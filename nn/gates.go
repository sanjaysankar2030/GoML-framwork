package nn

import (
	"fmt"
	"math"
	"math/rand"
)
	/* var train = [][3]int{
		{0, 0, 0},
		{0, 1, 0},
		{1, 0, 0},
		{1, 1, 1},
	}
// * AND GATE */

// *OR GATE
var train = [][3]int{
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 1},
}
/* 
//NAND GATE
var train = [][3]int{
	{0, 0, 1},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
}
 */

func init() {
	rand.Seed(69)
}

func sigmoidf(x float64) float64 {
	var sig_x float64 = 1 / (1 + math.Exp(-x))
	return sig_x
}
func rand_float() float64 {
	return rand.Float64()
}

// Computes the cost
func cost(w1 float64, w2 float64, b float64) float64 {
	var result float64 = 0
	for i := 0; i < len(train); i++ {
		x1 := float64(train[i][0])                  //the initial data feed to the model
		x2 := float64(train[i][1])                  //the initial data feed to the model
		var y float64 = sigmoidf(x1*w1 + x2*w2 + b) // Y is prediction or predicted output
		d := y - float64(train[i][2])
		result += d * d
	}
	final := result / float64(len(train))
	return final
}

func Gates() {
	fmt.Println(train)
	var eps float64 = 1e+2
	var rate float64 = 1e+2
	// Weight which is parameter which we tweek around with
	var w1 float64 = rand_float()* float64(10.0) 
	var w2 float64 = rand_float() *float64(10.0) 
	var b float64 = rand_float() 
	var d_cost float64 = 0
	var b_cost float64 = 0
	var bias float64 = 0

	for i := 0; i < 20000; i++ {
		var c float64 = cost(w1, w2, b)
		d_cost = (cost(w1+eps, w2, b) - c) / eps
		b_cost = (cost(w1, w2+eps, b) - c) / eps
		bias = (cost(w1, w2, b+eps) - c) / eps
		w1 -= rate * d_cost
		w2 -= rate * b_cost
		b -= rate * bias
		// fmt.Println("Cost: ", cost(w1, w2), "w1: ", w1, "w2: ", w2)
	}
	fmt.Println("------------------------")
	fmt.Println("THE WEIGHT 1", w1)
	fmt.Println("THE WEIGHT 2 ", w2)
	fmt.Println("----------------------------")
	fmt.Println("The Final cost ", cost(w1, w2, b))
	for i := 0.0; i < 2; i++ {
		for j := 0.0; j < 2; j++ {
			fmt.Println(i, "|", j, "|", math.Round(sigmoidf(i*w1+j*w2+b)))
			// fmt.Println(i, "|", j, "|", (sigmoidf(i*w1+j*w2+b)))
		}
	}
}
