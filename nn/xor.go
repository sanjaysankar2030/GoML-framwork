package nn

// import (
// 	"fmt"
// 	"math"
// 	"math/rand"
// )

type Xor struct {
	or_w1 float64
	or_w2 float64
	or_b float64
	nand_w1 float64
	nand_w2 float64
	nand_b float64
	and_w1 float64
	and_w2 float64
	and_b float64
}
func forward (xor *Xor,x float64,y float64)float64{
	var a float64 = sigmoidf((xor.or_w1 * x + xor.or_w2*y + xor.or_b)) 
	var b float64 = sigmoidf((xor.nand_w1 * x + xor.nand_w2*y + xor.nand_b)) 
	return sigmoidf((a*xor.and_w1 + b * xor.and_w2 + xor.and_b))
}



