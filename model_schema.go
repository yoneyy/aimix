package aimix

import "slices"

type ThinkingLevel string

func (t ThinkingLevel) String() string {
	return string(t)
}

const (
	ThinkingOff     ThinkingLevel = "off"
	ThinkingMinimal ThinkingLevel = "minimal"
	ThinkingLow     ThinkingLevel = "low"
	ThinkingMedium  ThinkingLevel = "medium"
	ThinkingHigh    ThinkingLevel = "high"
	ThinkingXhigh   ThinkingLevel = "xhigh"
	ThinkingMax     ThinkingLevel = "max"
)

var ThinkingLevels = []ThinkingLevel{
	ThinkingOff,
	ThinkingMinimal,
	ThinkingLow,
	ThinkingMedium,
	ThinkingHigh,
	ThinkingXhigh,
	ThinkingMax,
}

// ThinkingLevelMap maps unified thinking levels to provider/model-specific values.
// A missing key uses the provider default; an empty string marks the level as
// unsupported. xhigh and max are only supported when mapped explicitly.
type ThinkingLevelMap map[ThinkingLevel]string

// Lookup returns the mapped wire value and whether the level has an entry.
func (m ThinkingLevelMap) Lookup(level ThinkingLevel) (string, bool) {
	v, ok := m[level]
	return v, ok
}

// SupportedThinkingLevels lists the levels the model accepts, in ascending order.
// Models without reasoning only support ThinkingOff.
func SupportedThinkingLevels(model Model) []ThinkingLevel {
	if !model.Reasoning {
		return []ThinkingLevel{ThinkingOff}
	}
	var levels []ThinkingLevel
	for _, level := range ThinkingLevels {
		mapped, ok := model.ThinkingLevels.Lookup(level)
		if ok && mapped == "" {
			continue
		}
		if (level == ThinkingXhigh || level == ThinkingMax) && !ok {
			continue
		}
		levels = append(levels, level)
	}
	return levels
}

// ClampThinkingLevel returns level if the model supports it, otherwise the nearest
// supported level above it, otherwise the nearest one below it.
func ClampThinkingLevel(model Model, level ThinkingLevel) ThinkingLevel {
	available := SupportedThinkingLevels(model)
	if slices.Contains(available, level) {
		return level
	}
	fallback := ThinkingOff
	if len(available) > 0 {
		fallback = available[0]
	}
	requested := slices.Index(ThinkingLevels, level)
	if requested < 0 {
		return fallback
	}
	for i := requested; i < len(ThinkingLevels); i++ {
		if slices.Contains(available, ThinkingLevels[i]) {
			return ThinkingLevels[i]
		}
	}
	for i := requested - 1; i >= 0; i-- {
		if slices.Contains(available, ThinkingLevels[i]) {
			return ThinkingLevels[i]
		}
	}
	return fallback
}
