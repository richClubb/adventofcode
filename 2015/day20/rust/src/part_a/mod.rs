use std::collections::HashMap;

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

fn is_prime(value: &usize) -> bool {

    if value == &1 {
        return false;
    }

    if value == &2 {
        return true;
    }

    let mut index = value.clone() - 1;
    while index >= 2 {
        if value % index == 0 {
            return false;
        }

        index -= 1;
    }

    return true;
}

fn find_primes(target: &usize) -> Vec<usize> {

    let mut primes = Vec::<usize>::new();

    if target <= &1 {
        return primes;
    }

    for index in 2..=target.clone() {
        if is_prime(&index) {
            primes.push(index);
        }
    }

    return primes;
}

fn find_prime_factors(target: &usize, primes: &Vec<usize>) -> Option<HashMap<usize, usize>> {

    let mut prime_factors: HashMap<usize, usize> = HashMap::new();

    if target == &1 {
        return None;
    }

    if target == &2 {
        prime_factors.insert(2, 1);
        return Some(prime_factors);
    }

    let mut remainder = target.clone();

    while remainder != 1 {  
        for prime in primes {
            let val = remainder / prime;
            let val_rem = remainder % prime;

            if val_rem == 0 {
                prime_factors.entry(*prime).and_modify(|val| *val += 1).or_insert(1);
                remainder = val;
                break;
            }
        }
    }

    return Some(prime_factors);
}

fn sum_of_integers(target: &usize, primes: &Vec<usize>) -> usize {

    if target == &1 {
        return 1;
    }

    let mut overall_total = 1;

    let prime_factors = find_prime_factors(target, primes);

    for (prime, count) in prime_factors.unwrap() {
        let mut total = 0 as usize;

        for index in 0..=count {
            total += prime.pow(index as u32);
        }
        
        overall_total *= total;
    }

    return overall_total;
}

pub fn part_a(input: &usize)
{
    println!("Part A");

    let primes_target = (*input as f32).sqrt() as usize;

    let primes = find_primes(&primes_target);

    println!("found primes {primes:?}");

    let mut index = 1;
    loop {
        let result = sum_of_integers(&index, &primes);

        if &result >= input {
            break;
        }
        
        index += 1;
        println!("index {index}");
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

    #[test]
    fn test_is_prime() {
        assert_eq!(is_prime(&1), false);
        assert_eq!(is_prime(&2), true);
        assert_eq!(is_prime(&3), true);
        assert_eq!(is_prime(&4), false);
        assert_eq!(is_prime(&5), true);
        assert_eq!(is_prime(&6), false);
        assert_eq!(is_prime(&7), true);
        assert_eq!(is_prime(&8), false);
        assert_eq!(is_prime(&9), false);
        assert_eq!(is_prime(&10), false);
        assert_eq!(is_prime(&11), true);
    }

    #[test]
    fn test_find_primes() {
        assert_eq!(find_primes(&1), Vec::<usize>::new());
        assert_eq!(find_primes(&2), Vec::<usize>::from([2]));
        assert_eq!(find_primes(&3), Vec::<usize>::from([2, 3]));
        assert_eq!(find_primes(&4), Vec::<usize>::from([2, 3]));
        assert_eq!(find_primes(&5), Vec::<usize>::from([2, 3, 5]));
        assert_eq!(find_primes(&6), Vec::<usize>::from([2, 3, 5]));
        assert_eq!(find_primes(&7), Vec::<usize>::from([2, 3, 5, 7]));
        assert_eq!(find_primes(&8), Vec::<usize>::from([2, 3, 5, 7]));
        assert_eq!(find_primes(&9), Vec::<usize>::from([2, 3, 5, 7]));
        assert_eq!(find_primes(&10), Vec::<usize>::from([2, 3, 5, 7]));
        assert_eq!(find_primes(&11), Vec::<usize>::from([2, 3, 5, 7, 11]));
        assert_eq!(find_primes(&12), Vec::<usize>::from([2, 3, 5, 7, 11]));
    }

    #[test]
    fn test_find_prime_factors() {
        // assert_eq!(find_prime_factors(&1), None);
        // assert_eq!(find_prime_factors(&2), Some(HashMap::<usize, usize>::from([(2,1)])));
        // assert_eq!(find_prime_factors(&3), Some(HashMap::<usize, usize>::from([(3, 1)])));
        // assert_eq!(find_prime_factors(&3), None);
    }
}