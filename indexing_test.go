package ndarray

import (
	"errors"
	"reflect"
	"testing"
)

func TestMaskSelect(t *testing.T) {
	a, _ := FromData([]float64{1, 5, 3, 9, 2, 8}, 2, 3)
	two, _ := Full(3, 1)
	mask, _ := a.Greater(two) // a > 3 -> [[0,1,0],[1,0,1]]

	sel, err := a.MaskSelect(mask)
	if err != nil {
		t.Fatal(err)
	}
	// numpy a[a>3] -> [5, 9, 8] in row-major order.
	wantData(t, sel, []int{3}, []float64{5, 9, 8})

	// All-false mask -> empty 1-D array.
	z, _ := Zeros(2, 3)
	empty, _ := a.MaskSelect(z)
	wantData(t, empty, []int{0}, []float64{})

	// A mask that broadcasts to a larger shape than a is rejected.
	big, _ := Ones(2, 2, 3)
	if _, err := a.MaskSelect(big); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("MaskSelect enlarging mask err = %v", err)
	}
	// A non-broadcastable mask is rejected.
	bad, _ := Ones(4)
	if _, err := a.MaskSelect(bad); !errors.Is(err, ErrBroadcast) {
		t.Errorf("MaskSelect non-broadcastable err = %v", err)
	}
}

func TestNonzero(t *testing.T) {
	m, _ := FromData([]float64{0, 1, 0, 1, 0, 1}, 2, 3)
	nz := m.Nonzero()
	// np.flatnonzero -> [1, 3, 5].
	wantData(t, nz, []int{3}, []float64{1, 3, 5})

	all0, _ := Zeros(3)
	wantData(t, all0.Nonzero(), []int{0}, []float64{})
}

func TestTake(t *testing.T) {
	a, _ := FromData([]float64{10, 20, 30, 40, 50}, 5)
	got, err := a.Take(0, 2, 4, -1)
	if err != nil {
		t.Fatal(err)
	}
	wantData(t, got, []int{4}, []float64{10, 30, 50, 50})

	// Works on the flattened view of a multi-D array.
	m, _ := FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	g2, _ := m.Take(5, 0)
	if !reflect.DeepEqual(g2.materialize(), []float64{6, 1}) {
		t.Errorf("Take flat = %v", g2.materialize())
	}

	if _, err := a.Take(5); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("Take oob err = %v", err)
	}
	if _, err := a.Take(-6); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("Take neg-oob err = %v", err)
	}
}
