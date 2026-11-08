package bench

import (
	"os"
	"strings"
	"testing"

	"github.com/go-scikit/refbench"
)

func TestInv(t *testing.T) {
	p, err := refbench.PublicFuncs("..")
	if err != nil {
		t.Fatal(err)
	}
	os.WriteFile(os.Getenv("INV_OUT"), []byte(strings.Join(p, "\n")+"\n"), 0o644)
}
