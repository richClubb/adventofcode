package partb

import "testing"

func TestExtractSeedRanges_1(t *testing.T) {

	test_seed_ranges, success := extractSeedRangesFromString("seeds: 1 2 3 5 8 13 1000000 100000000 1000000000 6")

	if success == false {
		t.Error("Success should be true")
	}

	if test_seed_ranges == nil {
		t.Fatal("Should have produced seed ranges")
	}

	println(len(test_seed_ranges))
	if len(test_seed_ranges) != 5 {
		t.Fatal("Should have 5 seed ranges")
	}

	if test_seed_ranges[0].start != 1 || test_seed_ranges[0].size != 2 {
		t.Error("Element 0 should be start 1, size 2")
	}
	if test_seed_ranges[1].start != 3 || test_seed_ranges[1].size != 5 {
		t.Error("Element 1 should be start 3, size 5")
	}
	if test_seed_ranges[2].start != 8 || test_seed_ranges[2].size != 13 {
		t.Error("Element 2 should be start 8, size 13")
	}
	if test_seed_ranges[3].start != 1000000 || test_seed_ranges[3].size != 100000000 {
		t.Error("Element 3 should be start 1000000, size 100000000")
	}
	if test_seed_ranges[4].start != 1000000000 || test_seed_ranges[4].size != 6 {
		t.Error("Element 4 should be start 1000000000, size 6")
	}
}
