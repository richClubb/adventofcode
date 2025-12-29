package parta

import "testing"

func TestExtractSeeds_1(t *testing.T) {

	test_seeds, success := extractSeedsFromString("seeds: 1 2 3 5 8 13 1000000 100000000 1000000000")

	if success == false {
		t.Error("Success should be true")
	}

	if test_seeds == nil {
		t.Fatal("Should have produced test seeds")
	}

	if len(test_seeds) != 9 {
		t.Fatal("Should have 9 seeds")
	}

	if test_seeds[0] != 1 {
		t.Error("Element 1 should be 1")
	}
	if test_seeds[1] != 2 {
		t.Error("Element 1 should be 1")
	}
	if test_seeds[2] != 3 {
		t.Error("Element 1 should be 1")
	}
	if test_seeds[3] != 5 {
		t.Error("Element 1 should be 1")
	}
	if test_seeds[4] != 8 {
		t.Error("Element 1 should be 1")
	}
	if test_seeds[5] != 13 {
		t.Error("Element 1 should be 1")
	}
	if test_seeds[6] != 1000000 {
		t.Error("Element 1 should be 1")
	}
	if test_seeds[7] != 100000000 {
		t.Error("Element 1 should be 1")
	}
	if test_seeds[8] != 1000000000 {
		t.Error("Element 1 should be 1")
	}
}

func TestExtractSeeds_invalid_prefix(t *testing.T) {

	test_seeds, success := extractSeedsFromString("Seeds: 1 2 3 5 8 13 1000000 100000000 1000000000")

	if success == true {
		t.Error("Success should be false")
	}

	if test_seeds != nil {
		t.Fatal("Should have produced no test seeds")
	}
}

func TestExtractSeeds_invalid_seed(t *testing.T) {

	test_seeds, success := extractSeedsFromString("seeds: 1 foo 3 5 8 13 1000000 100000000 1000000000")

	if success == true {
		t.Error("Success should be false")
	}

	if test_seeds != nil {
		t.Fatal("Should have produced no test seeds")
	}
}
