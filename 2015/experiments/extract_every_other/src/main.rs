fn main() {
    let vec1: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6];
    println!("Vec 1: {:?}", vec1);

    let vec2: Vec<i32> = vec1.clone().iter().enumerate().filter(|&(i, _)| i % 2 == 0).map(|(_, e)| e).collect();
    println!("Vec 2: {:?}", vec2);

    let vec3: Vec<i32> = vec1.clone().into_iter().enumerate().filter(|&(i, _)| i % 2 != 0).map(|(_, e)| e).collect();
    println!("Vec 2: {:?}", vec3);
}
