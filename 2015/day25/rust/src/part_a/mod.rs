
pub fn part_a(target_row: &usize, target_column: &usize)
{
    println!("Part A");

    let mut last_value: u128 = 20151125;

    let mut curr_row: usize = 1;
    let mut curr_col: usize = 1;

    let mut max_row: usize = 1;

    loop {

        if curr_row == 1 {
            max_row += 1;
            curr_row = max_row;
            curr_col = 1;
        }
        else {
            curr_col += 1;
            curr_row -= 1;
        }

        last_value = (last_value * 252533) % 33554393;

        if (&curr_row == target_row) && (&curr_col == target_column)
        {
            println!("{curr_row} {curr_col}: {last_value}");
            break;
        }
    }

}