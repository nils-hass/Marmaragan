# Marmaragan
Diplom Thesis by Lucian McIntyre

## Description
Marmaragan attempts to leverage the power of LLM's to generate annotations for verifiable programs, for the SPARK2014 subset of ADA.


## Installation

### Set Environment Variable for OpenAPI Key

Add to .env file:
```OPENAI_API_KEY=<OpenAPI-Key>```

### Set Environment Variable for Gnatprove path

Add to .env file:
```GNATPROVE_EXECUTABLE_PATH="/path/to/.../Gnat/bin/gnatprove"```

 ### Install requirements.txt

```pip install -r requirements.txt```


## Run Application
Set appropriate variables in ```main.py```

Run ```python main.py``` and view results in benchmark log file

## Tools and Scripts

### Benchmark Generation

- **`scripts/create_benchmark_from_ada.py`** - Convert Ada/SPARK source files (.adb/.ads) to benchmark .txt format
  ```bash
  python scripts/create_benchmark_from_ada.py my_programs/ benchmarks/my-benchmark/
  ```

- **`scripts/generate_benchmark_variants.py`** - Automated generation of benchmark variants by removing pragmas
  ```bash
  python scripts/generate_benchmark_variants.py benchmarks/baseline-programs output/
  ```
  Generates 5 types of variants: all_pragmas, last_invariant_all_loops, one_assert, all_pragmas_one_loop, last_invariant_one_loop

### Testing and Analysis

- **`scripts/dry_run_benchmark.py`** - Test benchmark files by compiling and checking for mediums
- **`scripts/gen_benchmark.py`** - Generate SPARK files from benchmark .txt files
- **`scripts/gen_benchmark_diff.py`** - Calculate differences between benchmark versions

### Documentation

- **`BENCHMARK_GENERATION_GUIDE.md`** - Comprehensive guide for creating benchmarks from verified programs
- **`docs/BENCHMARK_VARIANT_GENERATION.md`** - Detailed documentation for the variant generation tool








