use regex::Regex;
use memoize::memoize;

use std::fs::File;
use std::io::{BufRead, BufReader};

use std::collections::HashMap;

#[derive(Debug, Clone, Eq, Hash, PartialEq)] 
struct Operation {
    input: String,
    output: String,
}

impl Operation {
    pub fn new(input: &str) -> Self{
        let operation_re = Regex::new(r"^(?<input>[0-9a-z]{0,}\s((AND)|(OR)|(LSHIFT)|(RSHIFT))\s[0-9a-z]{1,})\s->\s(?<output>[a-z]{1,})").unwrap();
        let assingment_re= Regex::new(r"^(?<input>[0-9a-z]{0,})\s->\s(?<output>[a-z]{1,})").unwrap();
        let not_re = Regex::new(r"^(?<input>NOT\s[0-9a-z]{1,})\s->\s(?<output>[a-z]{1,})").unwrap();

        let result = operation_re.captures(input).map(|cap| {
                let input = cap.name("input").unwrap().as_str();
                let output = cap.name("output").unwrap().as_str();
                (input, output)
            }
        );

        if result.is_some() {
            let input = result.unwrap().0;
            let output = result.unwrap().1;

            return Operation { input: String::from(input), output: String::from(output) };
        }

        let result = assingment_re.captures(input).map(|cap| {
                let input = cap.name("input").unwrap().as_str();
                let output = cap.name("output").unwrap().as_str();
                (input, output)
            }
        );

        if result.is_some() {
            let input = result.unwrap().0;
            let output = result.unwrap().1;

            return Operation { input: String::from(input), output: String::from(output) };
        }

        let result = not_re.captures(input).map(|cap| {
                let input = cap.name("input").unwrap().as_str();
                let output = cap.name("output").unwrap().as_str();
                (input, output)
            }
        );

        if result.is_some() {
            let input = result.unwrap().0;
            let output = result.unwrap().1;

            return Operation { input: String::from(input), output: String::from(output) };
        }

        return Operation { input: String::from(""), output: String::from("") };
    }
}

fn solve_assignment(input: String, map: &mut HashMap<String, Operation>) -> Option<u16> {

    match input.parse::<u16>() {
        Ok(val) => Some(val),
        Err(_) => solve_for(input, map),
    }
}

fn solve_not(input: String, map: &mut HashMap<String, Operation>) -> Option<u16> {

    let input = match input.parse::<u16>() {
        Ok(val) => val,
        Err(_) => match solve_for(input, map)  {
            Some(value) => value,
            None => return None
        }
    };

    return Some(!input);
}

fn solve_and(input1_start: String, input2_start: String, map: &mut HashMap<String, Operation>) -> Option<u16> {

    let input1 = match input1_start.parse::<u16>() {
        Ok(val) => val,
        Err(_) => {
                match solve_for(input1_start, map) {
                Some(val) => val,
                None => return None
            }
        },
    };

    let input2 = match input2_start.parse::<u16>() {
        Ok(val) => val,
        Err(_) => {
                match solve_for(input2_start, map) {
                Some(val) => val,
                None => return None
            }
        },
    };

    return Some(input1 & input2);
}

fn solve_or(input1_start: String, input2_start: String, map: &mut HashMap<String, Operation>) -> Option<u16> {

    let input1 = match input1_start.parse::<u16>() {
        Ok(val) => val,
        Err(_) => {
                match solve_for(input1_start, map) {
                Some(val) => val,
                None => return None
            }
        },
    };

    let input2 = match input2_start.parse::<u16>() {
        Ok(val) => val,
        Err(_) => {
                match solve_for(input2_start, map) {
                Some(val) => val,
                None => return None
            }
        },
    };

    return Some(input1 | input2);
}

fn solve_lshift(input_start: String, shift_start: String, map: &mut HashMap<String, Operation>) -> Option<u16> {

    let input = match input_start.parse::<u16>() {
        Ok(val) => val,
        Err(_) => {
                match solve_for(input_start, map) {
                Some(val) => val,
                None => return None
            }
        },
    };

    let shift = match shift_start.parse::<u16>() {
        Ok(val) => val,
        Err(_) => {
                match solve_for(shift_start, map) {
                Some(val) => val,
                None => return None
            }
        },
    };

    return Some(input << shift);
}


fn solve_rshift(input_start: String, shift_start: String, map: &mut HashMap<String, Operation>) -> Option<u16> {

    let input = match input_start.parse::<u16>() {
        Ok(val) => val,
        Err(_) => {
                match solve_for(input_start, map) {
                Some(val) => val,
                None => return None
            }
        },
    };

    let shift = match shift_start.parse::<u16>() {
        Ok(val) => val,
        Err(_) => {
                match solve_for(shift_start, map) {
                Some(val) => val,
                None => return None
            }
        },
    };

    return Some(input >> shift);
}

fn solve_for(output: String, map: &mut HashMap<String, Operation>) -> Option<u16> {

    let binding = map.clone();
    let operation = binding.get(&output);
    match operation.clone() {
        Some(operation) => {
            let result = solve_operation(operation, map);

            if result.is_some() {
                map.insert(output.clone(), Operation{input: result.unwrap().to_string(), output: String::from(output)});
            }

            result
        },
        None => {
            None
        },
    }
}

fn solve_operation(op: &Operation, map: &mut HashMap<String, Operation>) -> Option<u16> {

    let input = &op.input;

    if input.contains("AND") {
        let values: Vec<&str> = input.split(" ").collect();

        let input1 = String::from(values[0]);
        let input2 = String::from(values[2]);

        return solve_and(input1, input2, map);
    }

    if input.contains("OR") {
        let values: Vec<&str> = input.split(" ").collect();

        let input1 = String::from(values[0]);
        let input2 = String::from(values[2]);

        return solve_or(input1, input2, map);
    }

    if input.contains("NOT") {
        let values: Vec<&str> = input.split(" ").collect();

        let input = String::from(values[1]);

        return solve_not(input, map);
    }

    if input.contains("LSHIFT") {
        let values: Vec<&str> = input.split(" ").collect();

        let input = String::from(values[0]);
        let shift = String::from(values[2]);

        return solve_lshift(input, shift, map);
    }

    if input.contains("RSHIFT") {
        let values: Vec<&str> = input.split(" ").collect();

        let input = String::from(values[0]);
        let shift = String::from(values[2]);

        return solve_rshift(input, shift, map);
    }

    let values: Vec<&str> = input.split(" ").collect();
    let input = String::from(values[0]);
    
    return solve_assignment(input, map);
} 

pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut operations: HashMap<String, Operation> = HashMap::new();
    let mut outputs: Vec<String> = Vec::new();

    for line in buf_reader.lines() {
        let op = Operation::new(&line.unwrap());

        let key = op.output.clone();
        outputs.push(op.output.to_string());
        if !operations.contains_key(&key) {
            operations.insert(key, op);
        }
        else {
            operations.remove(&op.output);
        }
    }

    for key in &outputs { 
        let result = solve_for(String::from(key.clone()), &mut operations);
        println!("{} = {:?}", key, result);     
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_solve_not() {
        assert_eq!(
            solve_not(
                String::from("0"), 
                &mut HashMap::new()
            ), 
            Some(65535)
        );
        assert_eq!(
            solve_not(
                String::from("1"), 
                &mut HashMap::new()
            ), 
            Some(65534)
        );
        assert_eq!(
            solve_not(
                String::from("65535"), 
                &mut HashMap::new()
            ), 
            Some(0)
        );
        assert_eq!(
            solve_not(
                String::from("65534"), 
                &mut HashMap::new()
            ), 
            Some(1)
        );
    }

    #[test]
    pub fn test_solve_and() {
        assert_eq!(
            solve_and(
                String::from("0"),
                String::from("1"),
                &mut HashMap::new()
            ), 
            Some(0)
        );
        assert_eq!(
            solve_and(
                String::from("1"),
                String::from("1"),
                &mut HashMap::new()
            ), 
            Some(1)
        );
        assert_eq!(
            solve_and(
                String::from("2"),
                String::from("1"),
                &mut HashMap::new()
            ), 
            Some(0)
        );
        assert_eq!(
            solve_and(
                String::from("3"),
                String::from("1"),
                &mut HashMap::new()
            ), 
            Some(1)
        );
        
    }

    #[test]
    pub fn test_solve_or() {
        assert_eq!(
            solve_or(
                String::from("0"),
                String::from("1"),
                &mut HashMap::new()
            ), 
            Some(1)
        );
        assert_eq!(
            solve_or(
                String::from("0"),
                String::from("0"),
                &mut HashMap::new()
            ), 
            Some(0)
        );
        assert_eq!(
            solve_or(
                String::from("2"),
                String::from("1"),
                &mut HashMap::new()
            ), 
            Some(3)
        );
        
    }

    #[test]
    pub fn test_solve_lshift() {
        assert_eq!(
            solve_lshift(
                String::from("0"),
                String::from("1"),
                &mut HashMap::new()
            ), 
            Some(0)
        );
        assert_eq!(
            solve_lshift(
                String::from("1"),
                String::from("1"),
                &mut HashMap::new()
            ), 
            Some(2)
        );
        assert_eq!(
            solve_lshift(
                String::from("1"),
                String::from("2"),
                &mut HashMap::new()
            ), 
            Some(4)
        );
        assert_eq!(
            solve_lshift(
                String::from("1"),
                String::from("3"),
                &mut HashMap::new()
            ), 
            Some(8)
        );
        assert_eq!(
            solve_lshift(
                String::from("65535"),
                String::from("1"),
                &mut HashMap::new()
            ), 
            Some(65534)
        );
    }

    #[test]
    pub fn test_solve_rshift() {
        assert_eq!(
            solve_rshift(
                String::from("0"),
                String::from("1"),
                &mut HashMap::new()
            ), 
            Some(0)
        );
        assert_eq!(
            solve_rshift(
                String::from("1"),
                String::from("1"),
                &mut HashMap::new()
            ), 
            Some(0)
        );
        assert_eq!(
            solve_rshift(
                String::from("3"),
                String::from("2"),
                &mut HashMap::new()
            ), 
            Some(0)
        );
        assert_eq!(
            solve_rshift(
                String::from("65535"),
                String::from("1"),
                &mut HashMap::new()
            ), 
            Some(32767)
        );
    }

    #[test]
    pub fn test_solve_operation() {
        assert_eq!(
            solve_operation(
                &Operation{
                    input: String::from("1 AND 3"),
                    output: String::from("b")
                },
                &mut HashMap::new()
            ), 
            Some(1)
        );
        assert_eq!(
            solve_operation(
                &Operation{
                    input: String::from("1 OR 3"),
                    output: String::from("b")
                },
                &mut HashMap::new()
            ), 
            Some(3)
        );
        assert_eq!(
            solve_operation(
                &Operation{
                    input: String::from("NOT 1"),
                    output: String::from("b")
                },
                &mut HashMap::new()
            ), 
            Some(65534)
        );
        assert_eq!(
            solve_operation(
                &Operation{
                    input: String::from("1 LSHIFT 1"),
                    output: String::from("b")
                },
                &mut HashMap::new()
            ), 
            Some(2)
        ); 
        assert_eq!(
            solve_operation(
                &Operation{
                    input: String::from("1 RSHIFT 1"),
                    output: String::from("b")
                },
                &mut HashMap::new()
            ), 
            Some(0)
        ); 
        assert_eq!(
            solve_operation(
                &Operation{
                    input: String::from("1674"),
                    output: String::from("b")
                },
                &mut HashMap::new()
            ), 
            Some(1674)
        );
        assert_eq!(
            solve_operation(
                &Operation{
                    input: String::from("c"),
                    output: String::from("b")
                },
                &mut HashMap::new()
            ), 
            None
        );
    }

    #[test]
    pub fn test_solve_operation_recursive() {
        let mut map: HashMap<String, Operation> = HashMap::from([
            (String::from("a"), Operation { input: String::from("1"), output: String::from("a") }),
            (String::from("c"), Operation { input: String::from("2"), output: String::from("c") }),
            (String::from("d"), Operation { input: String::from("3"), output: String::from("d") })
        ]);
        assert_eq!(
            solve_operation(
                &Operation{
                    input: String::from("a AND 3"),
                    output: String::from("b")
                },
                &mut map
            ), 
            Some(1)
        );
        assert_eq!(
            solve_operation(
                &Operation{
                    input: String::from("e AND 3"),
                    output: String::from("b")
                },
                &mut map
            ), 
            None
        );
        assert_eq!(
            solve_operation(
                &Operation{
                    input: String::from("1 AND d"),
                    output: String::from("b")
                },
                &mut map
            ), 
            Some(1)
        );
        assert_eq!(
            solve_operation(
                &Operation{
                    input: String::from("a AND d"),
                    output: String::from("b")
                },
                &mut map
            ), 
            Some(1)
        );
        assert_eq!(
            solve_operation(
                &Operation{
                    input: String::from("a OR d"),
                    output: String::from("b")
                },
                &mut map
            ), 
            Some(3)
        );
        assert_eq!(
            solve_operation(
                &Operation{
                    input: String::from("NOT a"),
                    output: String::from("b")
                },
                &mut map
            ), 
            Some(65534)
        );

    }

    #[test]
    pub fn test_solve_assignment() {
        assert_eq!(
            solve_assignment(
                String::from("65535"),
                &mut HashMap::new()
            ), 
            Some(65535)
        );
        assert_eq!(
            solve_assignment(
                String::from("ab"),
                &mut HashMap::new()
            ), 
            None
        );  
    }

}