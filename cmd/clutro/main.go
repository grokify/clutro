// Command clutro clusters CLUTO-format matrices from the command line using
// CLUTO's flag vocabulary, so existing scripts port by changing the binary
// name. Flags are parsed with the standard library, which accepts both
// "-clmethod=rb" (CLUTO style) and "--clmethod rb".
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
)

const usage = `usage: clutro <command> [flags] <args>

commands:
  vcluster  cluster the rows of a CLUTO matrix file into k clusters
  evaluate  score a clustering file against class labels
  golden    re-cluster every corpus in a directory and compare with its reference

run "clutro <command> -h" for the command's flags.
`

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		if !errors.Is(err, flag.ErrHelp) {
			fmt.Fprintln(os.Stderr, "clutro:", err)
		}
		os.Exit(1)
	}
}

// run dispatches to a subcommand; it is the testable entry point.
func run(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		fmt.Fprint(stderr, usage)
		return errors.New("no command")
	}
	switch args[0] {
	case "vcluster":
		return runVcluster(args[1:], stdout, stderr)
	case "evaluate":
		return runEvaluate(args[1:], stdout, stderr)
	case "golden":
		return runGolden(args[1:], stdout, stderr)
	case "-h", "--help", "help":
		fmt.Fprint(stdout, usage)
		return nil
	}
	fmt.Fprint(stderr, usage)
	return fmt.Errorf("unknown command %q", args[0])
}
