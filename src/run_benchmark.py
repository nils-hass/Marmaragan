from time import sleep
from src.util import *
import shutil
import logging
import time
import traceback
import requests
import os
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from langchain_openai import ChatOpenAI

# LangChain moved core message/output types between packages across versions.
try:
    from langchain.schema import HumanMessage, BaseMessage
except ImportError:
    from langchain_core.messages import HumanMessage, BaseMessage

try:
    from langchain_core.outputs import ChatResult
except ImportError:
    from langchain.schema import ChatResult
from typing import List, Tuple




class run_benchmark:
    """
    This class is used to run the benchmark. It contains methods to run the benchmark with different configurations and to generate the results. 
    All results are logged to the benchmark_run.log file.
    
    Args:
            system_message (str): Instructions detailing the context for the language model
            benchmark_dir (str): The directory containing the benchmark files.
            gpt_model (str): The GPT model to use. Choice of "gpt-3.5-turbo-1106", "gpt-4-0125-preview" or "gpt-4-0613"
            benchmark_program_indices (List[int]): A list of the indices of the programs in the benchmark to run. Default is all programs in the benchmark
    """    
    
    def __init__(self, system_message: str, prompt: str, benchmark_dir: str, gpt_model: str, n_solutions: int, retries: int, with_medium_in_prompt: bool, benchmark_program_indices: List[int] = list(range(1, 17)), llm_backend: str = "api"):
        
        # Set the system message and gpt model
        self.system_message = system_message
        self.prompt = prompt
        self.benchmark_dir = benchmark_dir
        self.gpt_model = gpt_model
        self.n_solutions = n_solutions
        self.retries = retries
        self.with_medium_in_prompt = with_medium_in_prompt
        self.llm_backend = llm_backend
        
        
        # Retrieve the benchmark files
        self.benchmark_txt_files = retrieve_benchmark_files(benchmark_dir, benchmark_program_indices)
        print(self.benchmark_txt_files)
        
        
        # Name of the temporary directory to store the spark files
        self.tmp_benchmark_dir = "tmp_benchmark_dir"


        # Setup Logger
        self.logger = logging.getLogger('gen_1')
        self.logger.setLevel(logging.INFO)

        # Check if the logger already has handlers
        if not self.logger.handlers:
            # Create a file handler which logs even debug messages
            fh = logging.FileHandler('benchmark_run.log')
            fh.setLevel(logging.INFO)

            # Create a formatter and set the formatter for the handler.
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)

            # Add the handler to the logger
            self.logger.addHandler(fh)
        
        
        self.start_time = None;
        self.end_time = None;
        
        # f string newline fix
        self.nl = "\n"
        


    
    def run(self) -> None:
        """
        This class runs the benchmark
        """
        
        
        # Initialise the run and retrieve the benchmark files
        benchmark_files = self.init_run()
        
        # Programs that ran to completion, so that a run ending early can still be summarised
        self.programs_completed = 0
        aborted_with = None
        
        try:
            self.run_programs(benchmark_files)
        
        # An error reaching this point (an expired key, an exhausted quota, an interrupt) ends the
        # run, but whatever was measured before it is still summarised before the error propagates.
        except BaseException as exc:
            aborted_with = exc
            self.logger.error(
                f"\n-----------------------------------\nBenchmark run aborted after {self.programs_completed} of {len(benchmark_files)} programs\n{traceback.format_exc()}-----------------------------------\n")
            raise
        
        finally:
            self.end_run(self.results, self.programs_completed, len(benchmark_files), aborted_with)


    def run_programs(self, benchmark_files: List[Tuple[str, str]]) -> None:
        """
        Generate and prove the attempts for every program of the benchmark.
        
        Args:
            benchmark_files (List[Tuple[str, str]]): Tuples of the gpr filepath and the benchmark txt filepath
        
        Returns:
            None
        """
        
    # Iterate over each project in the benchmark
        for benchmark_file_path in benchmark_files:
            gpr_file_path = benchmark_file_path[0]
            benchmark_txt_path = benchmark_file_path[1]
            project_name = gpr_file_path.split('/')[-1]
            
            # Get dependencies and format as a string, get package_body
            dependencies = self.nl.join(retrieve_dependencies(benchmark_txt_path))
            package_body = retrieve_package_body(benchmark_txt_path)
            original_package_body = package_body
            
            # Format the prompt
            prompt_base = self.prompt.format(dependencies=dependencies, package_body=package_body)
            

            # Dict which keeps track of gnatprove output of each generation. 
            # Key is the response number, value is a tuple containing the llm generated code and gnatprove output
            self.gnatprove_output_dict = {}

            # Always run one initial proof pass. If no mediums remain, skip all LLM work.
            initial_gnatprove_output = run_gnatprove(gpr_file_path)
            if is_compilation_successful(initial_gnatprove_output):
                initial_mediums = parse_gnatprove_output(initial_gnatprove_output)
                if len(initial_mediums) == 0:
                    self.results.append(
                        (f"{project_name} {benchmark_txt_path} - attempt: 0 - retry: 0", True, True)
                    )
                    self.logger.info(
                        f"Solution found for {project_name} - attempt: 0 - retry: 0 (initial gnatprove run already medium-free)\n\n"
                    )
                    self.programs_completed += 1
                    continue

            print(benchmark_txt_path)

            # Create per-attempt isolated project directories to avoid file clobbering while pipelining.
            attempt_states = []
            for response_number_counter in range(1, self.n_solutions + 1):
                source_project_dir = "/".join(gpr_file_path.split("/")[:-1])
                source_project_name = gpr_file_path.split("/")[-1]
                source_project_dir_name = source_project_dir.split("/")[-1]
                attempt_dir = os.path.join(
                    self.tmp_benchmark_dir,
                    f"{source_project_dir_name}_attempt_{response_number_counter}",
                )

                if os.path.isdir(attempt_dir):
                    shutil.rmtree(attempt_dir)
                shutil.copytree(source_project_dir, attempt_dir)

                attempt_gpr_path = os.path.join(attempt_dir, source_project_name)

                if self.with_medium_in_prompt:
                    initial_prompt = extract_mediums(attempt_gpr_path, initial_gnatprove_output, prompt_base)
                else:
                    initial_prompt = prompt_base

                attempt_states.append({
                    "response_number": response_number_counter,
                    "retry_counter": 0,
                    "gpr_file_path": attempt_gpr_path,
                    "benchmark_file_name": benchmark_txt_path,
                    "dependencies": dependencies,
                    "next_prompt": initial_prompt,
                    "active": True,
                })

            llm_executor = ThreadPoolExecutor(max_workers=1)
            gnatprove_executor = ThreadPoolExecutor(max_workers=1)
            future_to_task = {}
            solution_found_flag = False

            try:
                # Seed one LLM call for each branch; workers serialize execution and pipeline with gnatprove.
                for state in attempt_states:
                    llm_future = llm_executor.submit(self.invoke_llm, state["next_prompt"], self.gpt_model, 1)
                    future_to_task[llm_future] = ("llm", state)

                while future_to_task and not solution_found_flag:
                    done_futures, _ = wait(list(future_to_task.keys()), return_when=FIRST_COMPLETED)

                    for completed_future in done_futures:
                        task_type, state = future_to_task.pop(completed_future)

                        if not state["active"]:
                            continue

                        if task_type == "llm":
                            llm_responses = completed_future.result()
                            llm_response = llm_responses[0]

                            proof_future = gnatprove_executor.submit(
                                self.extract_compile_and_log,
                                llm_response,
                                state["gpr_file_path"],
                                state["benchmark_file_name"],
                                state["response_number"],
                                state["retry_counter"],
                                original_package_body,
                            )
                            future_to_task[proof_future] = ("gnatprove", state)

                        else:
                            solution_found_flag, gnatprove_output_flag = completed_future.result()
                            if solution_found_flag:
                                break

                            if state["retry_counter"] < self.retries:
                                state["retry_counter"] += 1

                                response_number_counter = state["response_number"]
                                attempt_gpr_path = state["gpr_file_path"]

                                if response_number_counter in self.gnatprove_output_dict:
                                    # The attempt directory holds the code of the last response of this
                                    # branch that could be extracted, so that code is what the retry
                                    # prompt has to show.
                                    llm_code, gnatprove_output = self.gnatprove_output_dict[response_number_counter]
                                    retry_prompt = self.prompt.format(
                                        dependencies=state["dependencies"], package_body=llm_code
                                    )

                                    if gnatprove_output_flag:
                                        retry_prompt = extract_mediums(attempt_gpr_path, gnatprove_output, retry_prompt)
                                    else:
                                        # The current response could not be extracted, so the directory
                                        # still holds the code of an earlier iteration and has to be
                                        # proved again to describe what is on disk.
                                        retry_prompt = compile_and_append_stdout(attempt_gpr_path, retry_prompt)
                                else:
                                    # Neither the code nor the filename could ever be extracted for this
                                    # branch, so nothing was written to its attempt directory: it still
                                    # holds the original package body, which the initial proof output
                                    # describes exactly. Reusing that output also avoids a redundant
                                    # gnatprove run on the scheduler thread, which would stall the
                                    # pipeline for the duration of a full proof.
                                    retry_prompt = append_gnatprove_stdout(initial_gnatprove_output, prompt_base)

                                state["next_prompt"] = retry_prompt
                                next_llm_future = llm_executor.submit(
                                    self.invoke_llm, state["next_prompt"], self.gpt_model, 1
                                )
                                future_to_task[next_llm_future] = ("llm", state)
                            else:
                                state["active"] = False
            finally:
                for pending_future in future_to_task:
                    pending_future.cancel()

                llm_executor.shutdown(wait=True, cancel_futures=True)
                gnatprove_executor.shutdown(wait=True, cancel_futures=True)

            self.programs_completed += 1





    def init_run(self) -> List[Tuple[str, str]]:
        """
        This class initialises the run and creates the temporary files for the benchmark.
        
        Args:
            prompt_type (str): The type of prompting technique to use. Either "chain_of_thoughts" or "gnatprove_pre_compile"
        
        Returns:
            List[Tuple[str, str]]: A list of tuples containing the temporary filepath of the gpr files and the permanent txt files.
        
        """
        
        # self.logger.info(
        #     f"New Run \nGPT Model: {gpt_model} \nInstructions: \n{instructions} \n-----------------------------------\n\n")
        
        
        # Start timing
        self.start_time = time.time()
        print("Starting Benchmark Run:\n")
        
        # Pretty print the benchmark files
        benchmark_programs = [file.split("/")[-1] for file in self.benchmark_txt_files]
        
        
        self.logger.info(f"""
\n\n\n
--------------------------
Starting new Benchmark Run
--------------------------
LLM Backend: {self.llm_backend} \n
Model: {self.gpt_model} \n
Benchmark: {self.benchmark_dir} \n
Programs: \n{self.nl.join(map(str, benchmark_programs))} \n
n: {self.n_solutions}
Retries: {self.retries}
With Mediums in Prompt: {self.with_medium_in_prompt}\n
System Message: \n{self.system_message}
--------------------------
Prompt: \n{self.prompt}\n
--------------------------
\n\n\n
                         """)
        
        if self.with_medium_in_prompt:
            self.logger.info("""*** The prompt will include individually extracted mediums from the gnatprove output ***""")
        
        
        # Array of tuples, containing the results, of the form:
        # [(program-name, gnatprove-successful-compilation, medium-free)]
        self.results = []
        
        # Remove dir if it exists 
        try:
            shutil.rmtree(self.tmp_benchmark_dir)
            sleep(1)
        except Exception:
            pass
        
        # generate temporary directory for the spark files
        os.mkdir(self.tmp_benchmark_dir)
        
        # Array of containing the temporary filepath of the gpr files and the permanent txt files, as a tuple
        benchmark_files = []
        
        # Iterate over each file in the benchmark and generate the files in the spark_benchmark directory
        for benchmark_file_path in self.benchmark_txt_files:
            gpr_file_path = generate_spark_files(benchmark_file_path, self.tmp_benchmark_dir)
            benchmark_files.append((gpr_file_path, benchmark_file_path))
            
        
        return benchmark_files
        
    
    def end_run(self, results_array: List[Tuple[bool, bool]], programs_completed: int = None, programs_total: int = None, aborted_with: BaseException = None) -> None:
        """
        This class ends the run by sumarrising results in the log
        
        Args:
            results_array (List[Tuple[bool, bool]]): A list of tuples containing the results of the form [(program-name, gnatprove-successful-compilation, medium-free)]
            programs_completed (int): The number of programs that ran to completion
            programs_total (int): The number of programs the run set out to process
            aborted_with (BaseException): The error that ended the run early, if there was one
        
        Returns:
            None
        """
        
        successes = 0
        total = 0
        
        # Compile a string of the results_array
        summary_array = []
        for project, compilation_successful, no_mediums in results_array:
            if no_mediums:
                summary_array.append(f"Success: {project}")
                successes += 1
                total += 1
            else:
                summary_array.append(f"Failure: {project}")
                total += 1
                
        
        summary_string = "\n".join(summary_array)
        
        # Compile per-file statistics
        file_results = {}  # Maps file identifier to list of success/failure results
        
        for project, compilation_successful, no_mediums in results_array:
            # Extract file identifier (everything before " - attempt:")
            file_id = project.split(" - attempt:")[0] if " - attempt:" in project else project
            
            if file_id not in file_results:
                file_results[file_id] = []
            
            file_results[file_id].append(no_mediums)
        
        # Determine which files were successfully verified (at least one success)
        per_file_summary = []
        files_verified = 0
        total_files = len(file_results)
        
        for file_id in sorted(file_results.keys()):
            has_success = any(file_results[file_id])
            if has_success:
                per_file_summary.append(f"SUCCESS: {file_id}")
                files_verified += 1
            else:
                per_file_summary.append(f"FAILURE: {file_id}")
        
        per_file_summary_string = "\n".join(per_file_summary)
        
        # A summary of a run that ended early must not read like a complete one
        if aborted_with is not None:
            run_header = (
                f"Benchmark Run ABORTED after {programs_completed} of {programs_total} programs\n"
                f"{type(aborted_with).__name__}: {aborted_with}\n"
                f"The results below cover only the programs that finished."
            )
        elif programs_total is not None:
            run_header = f"End of Benchmark Run ({programs_completed} of {programs_total} programs)"
        else:
            run_header = "End of Benchmark Run"

        # End timing and log
        self.end_time = time.time()  # End timing
        duration = self.end_time - self.start_time

        self.logger.info(f"""
\n\n\n
--------------------------
{run_header}
--------------------------
{summary_string}
Time taken: {duration} \n
Summary of results (all attempts):
{successes} / {total}
--------------------------

Per-File Results:
{per_file_summary_string}

Files successfully verified (at least one success): {files_verified} / {total_files}
--------------------------
\n\n\n
                         """)
    



    def invoke_llm(self, prompt: str, model_name: str, n_solutions: int) -> List[str]:
        """
        This class invokes the LLM to generate n solutions for a given prompt. If n is 1, it returns a single solution
        
        Args:
            prompt (str): The prompt to send to the LLM
            model_name (str): The name of the model to use
            n_solutions (int): The number of solutions to generate
            
        Returns:
            List[str]: A list of strings containing each of the responses
        """
        
        # Log the prompt
        self.logger.info(f"\n-----------------------------------\nPrompt sent to LLM (backend={self.llm_backend}, model={model_name}, n={n_solutions}):\n{prompt}\n-----------------------------------\n")
        
        # List of solutions
        solutions = []
        
        # Create a human message
        message = HumanMessage(content=prompt)

        if self.llm_backend == "api":
            chat_model = ChatOpenAI(
                model_name=model_name,
                temperature=1.0,
                n=n_solutions
            )

            # Generate responses
            response = chat_model._generate([message])

            # Check if the response is a ChatResult object
            assert isinstance(response, ChatResult)
            assert len(response.generations) == n_solutions  # Check if correct number responses are generated
            for generation in response.generations:
                assert isinstance(generation.message, BaseMessage)
                assert isinstance(generation.message.content, str)
                solutions.append(generation.message.content)

        elif self.llm_backend == "local":
            # Import locally so API-only runs do not fail at module import time.
            chat_ollama_cls = None
            try:
                from langchain_community.chat_models import ChatOllama
                chat_ollama_cls = ChatOllama
            except ImportError:
                try:
                    from langchain_community.chat_models.ollama import ChatOllama
                    chat_ollama_cls = ChatOllama
                except ImportError:
                    try:
                        from langchain_ollama import ChatOllama
                        chat_ollama_cls = ChatOllama
                    except ImportError as exc:
                        self.logger.warning(
                            "Could not import ChatOllama integration (%s). Falling back to direct Ollama HTTP API.",
                            str(exc),
                        )

            if chat_ollama_cls is not None:
                chat_model = chat_ollama_cls(
                    model=model_name,
                    temperature=1.0
                )

                # ChatOllama may return one generation per call; loop for n solutions.
                for _ in range(n_solutions):
                    response = chat_model._generate([message])
                    assert isinstance(response, ChatResult)
                    assert len(response.generations) >= 1
                    generation = response.generations[0]
                    assert isinstance(generation.message, BaseMessage)
                    assert isinstance(generation.message.content, str)
                    solutions.append(generation.message.content)
            else:
                # Fallback path for offline environments without langchain ollama packages.
                for _ in range(n_solutions):
                    response = requests.post(
                        "http://127.0.0.1:11434/api/generate",
                        json={
                            "model": model_name,
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": 1.0},
                        },
                        timeout=600,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    generated_text = payload.get("response")
                    if not isinstance(generated_text, str):
                        raise RuntimeError(
                            f"Unexpected Ollama response format: {payload}"
                        )
                    solutions.append(generated_text)

        else:
            raise ValueError(f"Unsupported llm_backend: {self.llm_backend}. Expected 'api' or 'local'.")

        
        
        return solutions

    def extract_compile_and_log(self, llm_response: str, gpr_file_path: str, benchmark_file_name: str, response_number_counter: int, retry_counter: int, original_package_body: str) -> Tuple[bool, bool]:
        """
        This function extracts the code from the response, extracts the filepath and then overwrites the file in the temporary benchmark folder. 
        It then runs gnatprove on the project and logs the results. It returns a tuple indicating if a solution was found and if any mediums can be included in the retry.
        
        Args:
            llm_response (str): The response from the LLM
            gpr_file_path (str): The path to the gpr file
            benchmark_file_name (str): The name of the benchmark file
            response_number_counter (int): The index used to track the current llm response
            retry_counter (int): The index used to track the current retry number
            original_package_body (str): The original package body
        
        Returns:
            Tuple[bool, bool]: A tuple containing two boolean values. The first indicates if a solution was found. The second indicates if gnatprove output exists.
        """

        
        # Init variables
        project_name = gpr_file_path.split('/')[-1]
        compile_success = True
        llm_code = ""
        adb_file_path = ""

        # Extract the code from the response
        try:
            llm_code = extract_code_from_response(llm_response)

        # If the code cannot be extracted, set the compile success flag to False and log the error
        except ValueError as e:
            compile_success = False
            self.logger.error(
                f"\n-----------------------------------\nError extracting code from response number: {project_name} - benchmark No.: {benchmark_file_name} - attempt: {response_number_counter} - retry: {retry_counter}\nCode: {llm_response}\n-----------------------------------\n\n")

        # If code is not empty, extract the filename
        if llm_code not in ["", None]:

            # Extract the filename from the response
            try:
                adb_filename = extract_filename_from_response(
                    llm_code)

                # Convert filename to lowercase and add .adb extension
                filename_with_extension = adb_filename.lower() + ".adb"

                project_dir = "/".join(gpr_file_path.split("/")[:-1])
                adb_file_path = project_dir + "/" + filename_with_extension

                # Overwrite the destination file with the response code
                overwrite_destination_file_with_string(
                    adb_file_path, llm_code)

            # If the filename cannot be extracted, set the compile success flag to False and log the error
            except ValueError as e:
                compile_success = False
                self.logger.error(
                    f"\n-----------------------------------\n\nError extracting filename from response: {project_name} - benchmark No.: {benchmark_file_name} - attempt: {response_number_counter} - retry: {retry_counter}\nCode: {llm_code}\n-----------------------------------\n\n")

        # Three cases:
        # 1. Code was successfully extracted and gnatprove made it to stage 2. of compilation
        # 2. Code was successfully extracted but gnatprove did not make it to stage 2. of compilation
        # 3. Code was not successfully extracted and gnatprove did not run

        full_file_name = str(project_name) + " " + str(benchmark_file_name)

        if compile_success == True:

            # Run gnatprove on the project
            gnatprove_output = run_gnatprove(gpr_file_path)
            gnatprove_successful_compilation = is_compilation_successful(
                gnatprove_output)

            # Case 1
            # Code was successfully extracted and gnatprove made it to stage 2. of compilation
            if gnatprove_successful_compilation:

                # Parse the new mediums
                new_mediums = parse_gnatprove_output(gnatprove_output)

                self.results.append(
                    (f"{full_file_name} - attempt: {response_number_counter} - retry: {retry_counter}", compile_success, len(new_mediums) == 0))

                # Logging
                self.logger.info(
                    f"Project: {project_name} - benchmark No.: {benchmark_file_name} - attempt: {response_number_counter} - retry: {retry_counter} \n\nResponse: \n{self.nl.join(compute_diff(original_package_body, llm_code))}\n\nNew Mediums: \n{new_mediums}\n\nGnatprove Output: \n{gnatprove_output} \n-----------------------------------\n\n")

                # If the solution was medium free, break the loop
                if len(new_mediums) == 0:

                    # Log that a solution was found
                    self.logger.info(
                        f"Solution found for {project_name} - attempt: {response_number_counter} - retry: {retry_counter}\n\n")

                    return True, False
                
                # If not, continue
                else:
                    # Add the llm generated code and gnatprove output to the dictionary
                    self.gnatprove_output_dict[response_number_counter] = llm_code, gnatprove_output

                    return False, True
                    

            # Case 2
            # Code was successfully extracted but gnatprove did not make it to stage 2. of compilation
            else:
                self.results.append(
                    (f"{full_file_name} - attempt: {response_number_counter} - retry: {retry_counter}", compile_success, False))

                # Logging
                self.logger.info(
                    f"Project: {project_name} - benchmark No.: {benchmark_file_name} - attempt: {response_number_counter} - retry: {retry_counter} \n\nResponse: \n{self.nl.join(compute_diff(original_package_body, llm_code))}\n\nGnatprove Output: \n{gnatprove_output} \n-----------------------------------\n\n")
                
                # Add the llm generated code and gnatprove output to the dictionary
                self.gnatprove_output_dict[response_number_counter] = llm_code, gnatprove_output
                
                return False, True

        # Case 3
        #  Code was not successfully extracted and gnatprove did not run
        else:

            self.results.append(
                (f"{full_file_name} - attempt: {response_number_counter} - retry: {retry_counter}", False, False))

            # Logging
            self.logger.info(
                f"Project: {project_name} - benchmark No.: {benchmark_file_name} - attempt: {response_number_counter} - retry: {retry_counter}\nError: GnatProve did not run. Either the filename or code could not be extracted from the response.\n\nResponse: \n{llm_code}\n-----------------------------------\n\n")

            return False, False