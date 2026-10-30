package aimix

import "encoding/json"

// Tool declares a function the model may call. Executing tools is up to the caller.
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	// Parameters is the JSON Schema of the arguments object.
	Parameters json.RawMessage `json:"parameters"`
	// Strict asks the API to enforce Parameters exactly (strict mode / structured
	// outputs). The schema must then follow the provider's strict-mode rules, such
	// as "additionalProperties": false and every property listed in "required";
	// otherwise the request is rejected. False leaves validation to the caller.
	Strict bool `json:"strict,omitempty"`
}
