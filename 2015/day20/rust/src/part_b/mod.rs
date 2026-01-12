

fn find_multiples_calc_result(input: &usize) -> usize {

    let mut index = input.clone();
    let mut result = 0;

    while index > 0 {

        if input % index == 0 {

            let mult = input / (50 * index);
            let mult_rem = input % (50 * index);
            if ((mult == 1) && (mult_rem == 0)) || (mult < 1) {
                result += 11 * index;
            }
        }

        index -= 1;
    }

    return result;
}

pub fn part_b(input: &usize)
{
    println!("Part B");

    let mut index = 0;
    let mut max = 0 as usize;
    loop {
        let result = find_multiples_calc_result(&index);

        if result > max {
            max = result;
        }

        if (index % 10000) == 0 {
            println!("index: {index}, {max}");
        }

        if &result >= input {
            let one_minus = find_multiples_calc_result(&(index - 1));

            if one_minus > result {
                index = index - 1;
            }

            break;
        }

        index += 2;
    }

    println!("Result: {index}");
}


#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_find_multiples_calc_result() {
    //     assert_eq!(find_multiples_calc_result(&1), 11);
    //     assert_eq!(find_multiples_calc_result(&2), 33);
    //     assert_eq!(find_multiples_calc_result(&3), 44);
    //     assert_eq!(find_multiples_calc_result(&4), 77);
    //     assert_eq!(find_multiples_calc_result(&5), 66);
    //     assert_eq!(find_multiples_calc_result(&6), 132);
    //     assert_eq!(find_multiples_calc_result(&7), 88);
    //     assert_eq!(find_multiples_calc_result(&8), 165);
    //     assert_eq!(find_multiples_calc_result(&9), 143);
    //     assert_eq!(find_multiples_calc_result(&11), 132);
    //     assert_eq!(find_multiples_calc_result(&12), 297);
    //     assert_eq!(find_multiples_calc_result(&23), 253);
    //     assert_eq!(find_multiples_calc_result(&24), 627);
    // }
}