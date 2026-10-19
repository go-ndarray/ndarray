// Command regen runs every architecture's generator and moves its output into
// the kernels package: the native equivalent of the per-arch go:generate
// lines, which `go generate` only reaches for the host GOARCH.
//
// Run from internal/kernels/asmgen: go run ./regen
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

var arches = []string{"amd64", "arm64", "loong64", "ppc64le", "riscv64", "s390x"}

func main() {
	if err := run("."); err != nil {
		fmt.Fprintln(os.Stderr, "regen:", err)
		os.Exit(1)
	}
}

// run regenerates every architecture under root (the asmgen directory).
func run(root string) error {
	for _, a := range arches {
		dir := filepath.Join(root, a)
		cmd := exec.Command("go", "run", ".")
		cmd.Dir = dir
		cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("%s: %w", a, err)
		}
		out := "sum_" + a + ".s"
		if err := os.Rename(filepath.Join(dir, out), filepath.Join(root, "..", out)); err != nil {
			return fmt.Errorf("%s: %w", a, err)
		}
	}
	return nil
}
