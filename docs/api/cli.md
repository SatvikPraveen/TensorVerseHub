# Command-line interface

The package installs a single `tensorverse` command with sub-commands, plus
stand-alone aliases (`tensorverse-train`, `tensorverse-evaluate`,
`tensorverse-convert`, `tensorverse-serve`).

```bash
tensorverse --help
tensorverse info                                   # runtime / version report (JSON)

tensorverse train --task classification --architecture resnet --epochs 10 \
    --data ./data/images --image-size 64 64 --output-dir ./models --export-savedmodel

tensorverse evaluate --model ./models/final_model.keras --data ./data/test \
    --report --confusion-matrix --roc-curves --class-names cat dog

tensorverse convert --model ./models/final_model.keras --to tflite --quantize int8 --benchmark
tensorverse convert --model ./models/final_model.keras --to all

tensorverse serve --model ./models/final_model.keras --port 8000 --class-names cat dog
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"instances": [[[[0.1, 0.2, 0.3]]]]}'
```

Every command falls back to deterministic synthetic data when the `--data`
directory does not exist, so the whole pipeline can be exercised on a laptop:

```bash
tensorverse train --epochs 1 --image-size 32 32 --output-dir /tmp/tvh
tensorverse evaluate --model /tmp/tvh/final_model.keras --image-size 32 32 --report
tensorverse convert --model /tmp/tvh/final_model.keras --to tflite --quantize dynamic
```

## Python entry points

::: tensorversehub.cli
    options:
      heading_level: 3
      members: [main, build_parser]
