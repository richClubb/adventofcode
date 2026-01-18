use std::{collections::HashMap};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum Spell {
    MagicMissile,
    Drain,
    Shield,
    Poison,
    Recharge
}

fn available_spells(characters: Vec<&Character>) -> Vec<Spell> {

    let mut spells_active = Vec::<Spell>::new();

    for character in characters {
        for spell in &character.spells_active {
            spells_active.push(spell.0.clone());
        }
    }

    let mut spells_available = Vec::<Spell>::from([Spell::MagicMissile, Spell::Drain]);

    if !spells_active.contains(&Spell::Shield){
        spells_available.push(Spell::Shield);
    }

    if !spells_active.contains(&Spell::Recharge){
        spells_available.push(Spell::Recharge);
    }

    if !spells_active.contains(&Spell::Poison){
        spells_available.push(Spell::Poison);
    }

    return spells_available;
}

#[derive(Clone)]
struct Character {
    hit_points: isize,
    mana: isize,
    damage: usize,
    armour: usize,
    spells_active: HashMap<Spell, usize>,
}

impl Character {

    pub fn cast_spell(&mut self, target: &mut Character, spell: Spell) -> usize {
        // println!("Casting spell: '{spell:?}'");

        if spell == Spell::MagicMissile {
            target.hit_points -= 4;
            return 53;
        }

        if spell == Spell::Drain {
            self.hit_points += 2;
            target.hit_points -= 2;
            return 73;
        }

        if spell == Spell::Shield {
            self.spells_active.insert(Spell::Shield, 6);
            return 113;
        }

        if spell == Spell::Poison {
            target.spells_active.insert(Spell::Poison, 6);
            return 173;
        }

        if spell == Spell::Recharge {
            self.spells_active.insert(Spell::Recharge, 5);
            return 229;
        }

        return 0;
    }

    pub fn attack(&self, target: &mut Character) {
        target.hit_points -= self.damage as isize - target.armour as isize;
    }

    pub fn is_alive(&self) -> bool {
        if self.hit_points <= 0 {
            return false;
        }
        return true;
    }

    pub fn update_effects(&mut self) {

        for (spell, count ) in self.spells_active.clone() {
            if spell == Spell::Recharge {
                self.mana += 101;
            }

            if spell == Spell::Shield {
                self.armour = 7;
                if count == 1 {
                    self.armour = 0;
                }
            }

            if spell == Spell::Poison {
                self.hit_points -= 3;
            }

            if count == 1 {
                self.spells_active.remove(&spell);
            }
            else {
                self.spells_active.entry(spell).and_modify(|entry | {*entry -= 1});
            }
        }
    }
}



fn take_turn(hero: &mut Character, boss: &mut Character, mana_spent: usize) -> Option<usize> {

    //println!("Turn Hero: {} Boss: {}", hero.hit_points, boss.hit_points);
    hero.update_effects();
    boss.update_effects();

    if !boss.is_alive() {
        //println!("Boss dies, mana spent {mana_spent}");
        return Some(mana_spent);
    }

    boss.attack(hero);

    if !hero.is_alive(){
        //println!("Hero dead");
        return None;
    }

    hero.update_effects();
    boss.update_effects();

    if !boss.is_alive() {
        //println!("Boss dies");
        return Some(mana_spent);
    }

    let spells_available = available_spells(Vec::from([&hero.clone(), &boss.clone()]));
    //println!("spells available: {spells_available:?}");

    let mut min_mana = std::usize::MAX;
    for spell in spells_available {
        
        let mut hero = hero.clone();
        let mut boss = boss.clone();
        let mana = hero.cast_spell(&mut boss, spell);

        let result = take_turn(&mut hero, &mut boss, mana_spent + mana);
        if !boss.is_alive() {
            //println!("Boss dies");
            return Some(mana_spent);
        }

        match result {
            Some(result) => {
                if result < min_mana {
                    min_mana = result;
                }
            }
            None => ()
        }
    }

    return Some(min_mana);
}

pub fn part_a()
{
    println!("Part A");

    let mut hero = Character{hit_points: 50, mana: 500, damage: 0, armour: 0, spells_active: HashMap::new()};
    let mut boss = Character{hit_points: 71, mana: 0, damage: 10, armour: 0, spells_active: HashMap::new()};

    let spells_available = available_spells(Vec::from([&hero, &boss]));
    //println!("spells available: {spells_available:?}");

    let mut min_mana = std::usize::MAX;
    for spell in spells_available {

        //println!("Initial Hero: {} Boss: {}", hero.hit_points, boss.hit_points);

        let mut hero = hero.clone();
        let mut boss = boss.clone();
        let mana = hero.cast_spell(&mut boss, spell);

        let result = take_turn(&mut hero.clone(), &mut boss.clone(), mana);

        match result {
            Some(result) => {
                if result < min_mana {
                    min_mana = result;
                }
            }
            None => ()
        }
    }
}

