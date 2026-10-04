package ndarray_test

import (
	"fmt"

	"github.com/go-ndarray/ndarray"
)

func ExampleWorkspace() {
	state, _ := ndarray.FromData([]float64{1, 4, 9}, 3)
	w, _ := ndarray.FromData([]float64{2, 2, 2}, 3)
	bias, _ := ndarray.FromData([]float64{0, 1, 2}, 3)

	ws := ndarray.NewWorkspace()
	for step := 0; step < 3; step++ {
		x := ws.Use(state) // results computed from x come from ws
		p, _ := x.Mul(w)
		q, _ := p.Add(bias)
		state = q.Sqrt().Detach() // the one result that outlives the pass
		ws.Reset()                // everything else is recycled
	}
	fmt.Printf("%.4f %.4f %.4f\n", state.At(0), state.At(1), state.At(2))
	// Output: 1.8340 2.5083 2.9354
}
