
#[derive(Debug, PartialEq)]
pub enum Instruction {
    TurnOn,
    TurnOff,
    Toggle,
    Invalid
}

impl Instruction {
    pub fn new(input: String) -> Self {
        if input.contains("turn on") {
            return Instruction::TurnOn;
        }

        if input.contains("turn off") {
            return Instruction::TurnOff;
        }

        if input.contains("toggle") {
            return Instruction::Toggle;
        }

        return Instruction::Invalid;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_instruction() {
        assert_eq!(Instruction::new(&String::from("toggle")), Instruction::Toggle);
        assert_eq!(Instruction::new(&String::from("turn on")), Instruction::TurnOn);
        assert_eq!(Instruction::new(&String::from("turn off")), Instruction::TurnOff);
        assert_eq!(Instruction::new(&String::from("haggis")), Instruction::Invalid);
    }
}