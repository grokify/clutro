# Installation

## Library

```bash
go get github.com/grokify/clutro
```

```go
import (
    "github.com/grokify/clutro"           // Fit, options, Result
    "github.com/grokify/clutro/matrix"    // CSR, Dense, preprocessing
    "github.com/grokify/clutro/metric"    // Cosine, Euclidean
    "github.com/grokify/clutro/criterion" // I2
    "github.com/grokify/clutro/evaluate"  // ARI, NMI, silhouette, …
    "github.com/grokify/clutro/explain"   // feature descriptions
    "github.com/grokify/clutro/cluto"     // CLUTO file formats
)
```

CLUTRO depends only on the Go standard library — the library and the CLI
both — and builds with `CGO_ENABLED=0`. Go 1.26 or later is required.

## CLI

```bash
go install github.com/grokify/clutro/cmd/clutro@latest
clutro -h
```

## Examples

The `examples/` directory holds runnable programs that compile in CI:
`sparse` (the README example, with explanations), `dense` (embeddings with
spherical k-means and internal metrics), `treecut` (choosing k after the
fit), `cluto` (migrating a CLUTO workflow), and `hdbscan` (a separate
module feeding CLUTRO data to a density-based clusterer).

```bash
go run github.com/grokify/clutro/examples/sparse@latest
```
