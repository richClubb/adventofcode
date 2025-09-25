unsafe fn a_func(value: *mut u64) -> bool{

    if *value > 5 {
        *value = 9;
        return true
    }

    return false
}


fn main() {
    let mut value = 1;

    let result = unsafe { a_func(&mut value) };
    println!{"Result {} {}", value, result};

    value = 5;
    let result = unsafe { a_func(&mut value) };
    println!{"Result {} {}", value, result};

    value = 6;
    let result = unsafe { a_func(&mut value) };
    println!{"Result {} {}", value, result};
}
