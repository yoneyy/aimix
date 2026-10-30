package aimix

import "encoding/json"

// Tool declares a function the model may call. Executing tools is up to the caller.
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	// Parameters is the JSON Schema of the arguments object.
	Parameters json.RawMessage `json:"parameters"`
}
