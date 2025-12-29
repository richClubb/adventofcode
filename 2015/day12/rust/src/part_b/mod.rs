use std::fs;
use serde_json::{Result, Value, Number};

fn process_json(json_value: &Value) -> i32 {

    let mut result = 0;
    if json_value.is_object() {
        let json_object = json_value.as_object().unwrap();

        for (_, sub_json_value) in json_object {
            if sub_json_value.is_string() {
                let json_string = sub_json_value.as_str().unwrap();
                if json_string == "red" {
                    return 0;
                }
            }
        }

        for (_, sub_json_value) in json_object {
            result += process_json(sub_json_value);
        }


        return result;
    }

    if json_value.is_array() {
        let entries: &Vec<Value> = json_value.as_array().unwrap();
        for entry in entries {
            result += process_json(entry);
        }
        return result;
    }

    if json_value.is_number() {
        let val = json_value.as_number().unwrap();
        return val.as_i64().unwrap() as i32;
    }
    
    return 0;
}



pub fn part_b(path: &String)
{
    println!("Part B");

    let json_string = fs::read_to_string(path);

    let json_data: Value = serde_json::from_str(json_string.unwrap().as_str()).unwrap();

    let result = process_json(&json_data);

    println!("{result}");

}