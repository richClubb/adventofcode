use std::{collections::HashMap, mem::take};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum Spell {
    MagicMissile,
    Drain,
    Shield,
    Poison,
    Recharge,
}

impl Spell {
    pub fn cast(&self, caster: &mut Character, target: &mut Character, log: &mut Vec<String>) {
        
        caster.mana -= self.cost();

        match self {
            Spell::MagicMissile => {
                target.hit_points -= 4;
                log.push(format!("Magic missile cast: {caster:?} {target:?}"));
            },
            Spell::Drain => {
                caster.hit_points += 2;
                target.hit_points -= 2;
                log.push(format!("Drain cast: {caster:?} {target:?}"));
            },
            Spell::Shield => {
                caster.armour = 7;
                caster.spells_active.insert(Spell::Shield, 6);
                log.push(format!("Shield cast: {caster:?} {target:?}"));
            },
            Spell::Poison => {
                target.spells_active.insert(Spell::Poison, 6);
                log.push(format!("Poison cast: {caster:?} {target:?}"));
            },
            Spell::Recharge => {
                caster.spells_active.insert(Spell::Recharge, 5);
                log.push(format!("Recharge cast: {caster:?} {target:?}"));
            },
        }
    }

    pub fn cost(&self) -> isize {
        match self {
            Spell::MagicMissile => 53,
            Spell::Drain => 73,
            Spell::Shield => 113,
            Spell::Poison => 173,
            Spell::Recharge => 229,
        }
    }

    pub fn available_spells(mana_available: &isize, characters: Vec<&Character>) -> Option<Vec<Spell>> {

        if mana_available < &53 {
            return None
        }

        let mut spells_active = Vec::<Spell>::new();

        for character in characters {
            for (spell, _) in &character.spells_active {
                spells_active.push(spell.clone());
            }
        }

        let mut spells_available = Vec::<Spell>::new();

        if mana_available >= &Spell::MagicMissile.cost() {
            spells_available.push(Spell::MagicMissile);
        }

        if mana_available >= &Spell::Drain.cost() {
            spells_available.push(Spell::Drain);
        }

        if (mana_available >= &Spell::Shield.cost()) && !spells_active.contains(&Spell::Shield) {
            spells_available.push(Spell::Shield)
        }

        if (mana_available >= &Spell::Poison.cost()) && !spells_active.contains(&Spell::Poison) {
            spells_available.push(Spell::Poison)
        }

        if (mana_available >= &Spell::Recharge.cost()) && !spells_active.contains(&Spell::Recharge) {
            spells_available.push(Spell::Recharge)
        }

        return Some(spells_available);
    }
}

#[derive(Clone, Debug)]
struct Character {
    hit_points: isize,
    mana: isize,
    damage: isize,
    armour: isize,
    spells_active: HashMap<Spell, usize>
}

impl Character {
    pub fn check_effects(&mut self, log: &mut Vec<String>) {
        for (spell, duration) in self.spells_active.clone() {

            match spell {
                Spell::Shield => {
                    if duration <= 1 {
                        self.armour = 0;
                        log.push("Shield depleted".to_string());
                    }
                    else {
                        self.armour = 7;
                        log.push("Shield active".to_string());
                    }
                },
                Spell::Poison => {
                    self.hit_points -= 3;
                },
                Spell::Recharge => {
                    self.mana += 101;
                },
                _ => ()
            }

            if duration <= 1 {
                // println!("  {spell:?} expired");
                log.push(format!("{spell:?} expired"));
                self.spells_active.remove(&spell);
            }
            else {
                // println!("  {spell:?} now {}", duration - 1);
                log.push(format!("{spell:?} duration {}", duration - 1));
                self.spells_active.entry(spell).and_modify(|entry| *entry -= 1);
            }

        }
    }

    pub fn attack(&self, target: &mut Character, log: &mut Vec<String>) {
        target.hit_points -= (self.damage - target.armour).max(1);
        log.push(format!("damaged {} {target:?}", self.damage - target.armour).to_string());
    }

    pub fn is_dead(&self) -> bool {
        if self.hit_points <= 0 {
            return true;
        }
        return false;
    }
}

fn take_turn(hero: &mut Character, boss: &mut Character, spell_list: &mut Vec<Spell>, log: &mut Vec<String>) -> Option<isize> {

    log.push("Boss turn".to_string());
    // println!("Taking turn Hero {:?}, boss: {:?}", hero, boss);
    hero.check_effects(log);
    boss.check_effects(log);
    // println!("  Boss effect round: Hero {:?}, boss: {:?}", hero, boss);

    if boss.is_dead() {
        let mana_spent = spell_list.iter().fold(0, |acc, spell| acc + spell.cost());
        // println!("  Boss is dead, {spell_list:?}, {mana_spent}, {hero:?} {boss:?}");
        log.push(format!("  Boss is dead, {spell_list:?}, {mana_spent}, {hero:?} {boss:?}").to_string());
        // println!("{log:?}");
        return Some(mana_spent);
    }

    boss.attack(hero, log);
    // println!("  Boss action round: Hero {:?}, boss: {:?}", hero, boss);

    if hero.is_dead() {
        // println!("  Hero dead");
        return None;
    }

    log.push("Player turn".to_string());
    hero.check_effects(log);
    boss.check_effects(log);
    // println!("  Player effect round: Hero {:?}, boss: {:?}", hero, boss);

    if boss.is_dead() {
        let mana_spent = spell_list.iter().fold(0, |acc, spell| acc + spell.cost());
        // println!("  Boss is dead, {spell_list:?}, {mana_spent}, {hero:?} {boss:?}");
        log.push(format!("  Boss is dead, {spell_list:?}, {mana_spent}, {hero:?} {boss:?}").to_string());
        // println!("{log:?}");
        return Some(mana_spent);
    }

    let spells_available = Spell::available_spells(&hero.mana, Vec::from([&hero.clone(), &boss.clone()]));
    // println!("  Spells avilable: {:?}", spells_available);

    let mut min_mana = std::isize::MAX;
    let _ = match spells_available {
        Some(spells) => {
            for spell in spells {
                // println!("  \nCasting {:?}", spell);
                let mut spells_cast = spell_list.clone();
                spells_cast.push(spell.clone());
                let mut hero = hero.clone();
                let mut boss = boss.clone();
                let mut log = log.clone();

                // println!("Spells cast: {spells_cast:?}");
                spell.cast(&mut hero, &mut boss, &mut log);

                if boss.is_dead() {
                    let mana_spent = spells_cast.iter().fold(0, |acc, spell| acc + spell.cost());
                    //println!("  Boss is dead, {spells_cast:?}, {mana_spent}, {hero:?} {boss:?}");
                    log.push(format!("  Boss is dead, {spells_cast:?}, {mana_spent}, {hero:?} {boss:?}").to_string());
                    //println!("{log:?}");
                    if mana_spent < min_mana {
                        min_mana = mana_spent;
                        continue;
                    }
                }

                // println!("{hero:?} {boss:?}");
                let result = take_turn(&mut hero, &mut boss, &mut spells_cast, &mut log);
                if result.is_some() {
                    if result.unwrap() < min_mana {
                        min_mana = result.unwrap();
                    }
                }
            }
        }
        None => {
            // ()
            // println!("  Can't cast :( {hero:?} , {boss:?}");
            let result = take_turn(hero, boss, &mut spell_list.clone(), &mut log.clone());
            if result.is_some() {
                if result.unwrap() < min_mana {
                    min_mana = result.unwrap();
                }
            }
        },
    };

    return Some(min_mana);
}

pub fn part_a()
{
    println!("Part A");

    let mut hero = Character{ hit_points: 50, mana: 500, damage: 0, armour: 0, spells_active: HashMap::new()};
    let mut boss = Character{ hit_points: 71, mana: 0, damage: 10, armour: 0, spells_active: HashMap::new()};

    // let mut hero = Character{ hit_points: 10, mana: 250, damage: 0, armour: 0, spells_active: HashMap::new()};
    // let mut boss = Character{ hit_points: 13, mana: 0, damage: 8, armour: 0, spells_active: HashMap::new()};
    let spell_available = Spell::available_spells(&hero.mana, Vec::from([&hero, &boss]));

    let mut spells_cast = Vec::<Spell>::new();
    let mut log = Vec::<String>::new();

    let mut min_mana = std::isize::MAX;
    for spell in spell_available.unwrap() {
        // println!("{hero:?} {boss:?}");
        let mut spells_cast = spells_cast.clone();
        spells_cast.push(spell.clone());
        let mut hero = hero.clone();
        let mut boss = boss.clone();
        let mut log = log.clone();

        log.push("Player turn".to_string());
        // println!("Spells cast: {spells_cast:?}");
        spell.cast(&mut hero, &mut boss, &mut log);

        let result = take_turn(&mut hero, &mut boss, &mut spells_cast, &mut log.clone());
        if result.is_some() {
            if result.unwrap() < min_mana {
                min_mana = result.unwrap();
            }
        }
    }

    println!("Min mana: {min_mana}");
}
