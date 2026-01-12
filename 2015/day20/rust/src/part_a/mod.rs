

fn find_multiples_calc_result(input: &usize) -> usize {

    let mut index = input.clone();
    let mut result = 0;
    while index > 0 {
        if input % index == 0 {
            result += 10 * index;
        }

        index -= 1;
    }

    return result;
}

pub fn part_a(input: &usize)
{
    println!("Part A");

    let mut index = 1;
    loop {
        let result = find_multiples_calc_result(&index);

        if &result >= input {
            break;
        }

        index += 1;
    }

    println!("Result: {index}");
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_multiples_calc_result() {
        assert_eq!(find_multiples_calc_result(&1), 10);
        assert_eq!(find_multiples_calc_result(&2), 30);
        assert_eq!(find_multiples_calc_result(&3), 40);
        assert_eq!(find_multiples_calc_result(&4), 70);
        assert_eq!(find_multiples_calc_result(&5), 60);
        assert_eq!(find_multiples_calc_result(&6), 120);
        assert_eq!(find_multiples_calc_result(&7), 80);
        assert_eq!(find_multiples_calc_result(&8), 150);
        assert_eq!(find_multiples_calc_result(&9), 130);
        assert_eq!(find_multiples_calc_result(&11), 120);
        assert_eq!(find_multiples_calc_result(&12), 280);
    }
}