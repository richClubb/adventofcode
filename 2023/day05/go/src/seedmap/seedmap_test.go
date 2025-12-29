package seedmap

import "testing"

func TestNewSeedMap(t *testing.T) {
	sm := SeedMapNew(1, 10, 5)

	if sm.SourceStart != 1 {
		t.Error("Invalid source start")
	}

	if sm.SourceEnd != 6 {
		t.Error("Invalid source end")
	}

	if sm.TargetStart != 10 {
		t.Error("Invalid source end")
	}

	if sm.TargetEnd != 15 {
		t.Error("Invalid source end")
	}

	if sm.Size != 5 {
		t.Error("Invalid source end")
	}
}

func TestNewSeedMapFromString(t *testing.T) {
	sm, success := SeedMapNewFromString("10 1 5")

	if !success {
		t.Error("Could not parse string")
	}

	if sm.SourceStart != 1 {
		t.Error("Invalid source start")
	}

	if sm.SourceEnd != 6 {
		t.Error("Invalid source end")
	}

	if sm.TargetStart != 10 {
		t.Error("Invalid source end")
	}

	if sm.TargetEnd != 15 {
		t.Error("Invalid source end")
	}

	if sm.Size != 5 {
		t.Error("Invalid source end")
	}
}

func TestMapSeed_out_of_range_low(t *testing.T) {

	sm := SeedMapNew(2, 10, 5)

	value := uint64(1)
	new_value, result := sm.MapSeed(value)
	if result != false {
		t.Error("Error mapping inital value")
	}

	if new_value != 1 {
		t.Error("Error mapping value")
	}
}

func TestMapSeed_out_of_range_high(t *testing.T) {

	sm := SeedMapNew(2, 10, 5)

	value := uint64(7)
	new_value, result := sm.MapSeed(value)
	if result != false {
		t.Error("Error mapping inital value")
	}

	if new_value != 7 {
		t.Error("Error mapping value")
	}
}

func TestMapSeed_in_range_1(t *testing.T) {

	sm := SeedMapNew(2, 10, 5)

	value := uint64(2)
	new_value, result := sm.MapSeed(value)
	if result != true {
		t.Error("Error mapping inital value")
	}

	if new_value != 10 {
		t.Error("Error mapping value")
	}
}

func TestMapSeed_in_range_2(t *testing.T) {

	sm := SeedMapNew(2, 10, 5)

	value := uint64(3)
	new_value, result := sm.MapSeed(value)
	if result != true {
		t.Error("Error mapping inital value")
	}

	if new_value != 11 {
		t.Error("Error mapping value")
	}
}

func TestMapSeed_in_range_3(t *testing.T) {

	sm := SeedMapNew(2, 10, 5)

	value := uint64(6)
	new_value, result := sm.MapSeed(value)
	if result != true {
		t.Error("Error mapping inital value")
	}

	if new_value != 14 {
		t.Error("Error mapping value")
	}
}
