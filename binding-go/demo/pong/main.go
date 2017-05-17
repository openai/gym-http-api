package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"

	gym "github.com/openai/gym-http-api/binding-go"
)

const BaseURL = "http://localhost:5000"

func main() {
	client, err := gym.NewClient(BaseURL)
	must(err)

	// Create environment instance.
	id, err := client.Create("Pong-v0")
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		fmt.Fprintln(os.Stderr, "You might have to run `pip install gym[atari]`.")
		os.Exit(1)
	}
	defer client.Close(id)

	// Take a few random steps
	_, err = client.Reset(id)
	must(err)
	var lastObservation interface{}
	for i := 0; i < 5; i++ {
		action, err := client.SampleAction(id)
		must(err)
		lastObservation, _, _, _, err = client.Step(id, action, false)
		must(err)
	}

	// Produce an image from the last video frame and
	// save it to pong.png.
	frame := lastObservation.([][][]float64)
	img := image.NewRGBA(image.Rect(0, 0, len(frame[0]), len(frame)))
	for rowIdx, row := range frame {
		for colIdx, col := range row {
			color := color.RGBA{
				R: uint8(col[0]),
				G: uint8(col[1]),
				B: uint8(col[2]),
				A: 0xff,
			}
			img.SetRGBA(colIdx, rowIdx, color)
		}
	}
	outFile, err := os.Create("pong.png")
	must(err)
	defer outFile.Close()
	must(png.Encode(outFile, img))
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
