use std::fs;

#[test]
fn test_part_a() {
    let part_a_sample_file_path = "../part_a_sample.txt";

    let contents = fs::read_to_string(part_a_sample_file_path)
        .expect("Should have been able to read the file");
    let result = part_a::part_a(contents.clone());

    assert_eq!(142, 142);
}
