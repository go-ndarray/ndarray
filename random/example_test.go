package random_test

import (
	"fmt"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
)

// The numbers below are NumPy's, for the same seed:
//
//	rng = np.random.default_rng(42)
//	rng.random(3); rng.integers(0, 10, 5); rng.standard_normal(2)
func Example() {
	rng := random.DefaultRNG(42)
	u, _ := rng.Random(3)
	i, _ := rng.Integers(0, 10, 5)
	z, _ := rng.StandardNormal(2)
	fu, _ := ndarray.Data[float64](u)
	fi, _ := ndarray.Data[int64](i)
	fz, _ := ndarray.Data[float64](z)
	fmt.Println(fu)
	fmt.Println(fi)
	fmt.Println(fz)
	// Output:
	// [0.7739560485559633 0.4388784397520523 0.8585979199113825]
	// [0 6 2 0 5]
	// [0.12784040316728537 -0.3162425923435822]
}

// The legacy API: np.random.seed(0); np.random.rand(3); np.random.randn(2);
// np.random.randint(0, 10, 5).
func ExampleSeed() {
	random.Seed(0)
	u, _ := random.Rand(3)
	z, _ := random.Randn(2)
	i, _ := random.Randint(0, 10, 5)
	fu, _ := ndarray.Data[float64](u)
	fz, _ := ndarray.Data[float64](z)
	fi, _ := ndarray.Data[int64](i)
	fmt.Println(fu)
	fmt.Println(fz)
	fmt.Println(fi)
	// Output:
	// [0.5488135039273248 0.7151893663724195 0.6027633760716439]
	// [-2.268328201180374 1.3335453816218967]
	// [5 2 4 7 6]
}

// rng.choice(5, 3, replace=False)
func ExampleGenerator_ChoiceN() {
	idx, _ := random.DefaultRNG(42).ChoiceN(5, random.ChoiceOptions{NoReplace: true}, 3)
	d, _ := ndarray.Data[int64](idx)
	fmt.Println(d)
	// Output: [4 0 3]
}
