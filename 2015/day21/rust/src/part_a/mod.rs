#[derive(Clone, Debug)]
struct Equipment {
    name: String,
    cost: usize,
    attack: usize,
    defence: usize
}

struct Character {
    hit_points: isize,
    armour_val: usize,
    attack_val: usize,
}

fn get_all_equip_combinations(weapons: Vec<Equipment>, armour: Vec<Equipment>, rings: Vec<Equipment>) -> Vec<(Equipment, Option<Equipment>, Option<Vec<Equipment>>, usize)>
{

    let mut combinations = Vec::<(Equipment, Option<Equipment>, Option<Vec<Equipment>>, usize)>::new();

    for weapon in weapons {
        
        // add the weapon-only combination
        combinations.push((weapon.clone(), None, None, weapon.cost));

        // single ring combinations
        for ring in &rings {
            let cost = weapon.cost + ring.cost;
            combinations.push((weapon.clone(), None, Some(Vec::from([ring.clone()])), cost));
        }

        // dual ring combinations
        for (index, primary_ring) in rings.iter().enumerate() {
            
            let mut remaining_rings = rings.clone();
            remaining_rings.remove(index);

            for secondary_ring in remaining_rings {
                let cost = weapon.cost + primary_ring.cost + secondary_ring.cost;
                let selected_rings = Vec::from([primary_ring.clone(), secondary_ring]);
                combinations.push((weapon.clone(), None, Some(selected_rings.clone()), cost));
            }
        }

        // add in the weapon and armour but no ring combinations
        for armour_item in &armour {
            let cost = weapon.cost + armour_item.cost;
            combinations.push((weapon.clone(), Some(armour_item.clone()), None, cost));

            // single ring combinations
            for ring in &rings {
                let cost = cost + ring.cost;
                combinations.push((weapon.clone(), Some(armour_item.clone()), Some(Vec::from([ring.clone()])), cost));
            }

            // dual ring combinations
            for (index, primary_ring) in rings.iter().enumerate() {
                
                let mut remaining_rings = rings.clone();
                remaining_rings.remove(index);

                for secondary_ring in remaining_rings {
                    let cost = cost + primary_ring.cost + secondary_ring.cost;
                    let selected_rings = Vec::from([primary_ring.clone(), secondary_ring]);
                    combinations.push((weapon.clone(), Some(armour_item.clone()), Some(selected_rings.clone()), cost));
                }
            }
        }

    }

    return combinations;
}

pub fn part_a()
{
    println!("Part A");

    let weapons = Vec::from(
        [
            Equipment{name: String::from("Dagger"),     cost:   8, attack: 4, defence: 0},
            Equipment{name: String::from("Shortsword"), cost:  10, attack: 5, defence: 0},
            Equipment{name: String::from("Warhammer"),  cost:  25, attack: 6, defence: 0},
            Equipment{name: String::from("Longsword"),  cost:  40, attack: 7, defence: 0},
            Equipment{name: String::from("Greataxe"),   cost:  74, attack: 8, defence: 0},
        ]
    );

    let armour = Vec::from(
        [
            Equipment{name: String::from("Leather"),    cost:  13, attack: 0, defence: 1},
            Equipment{name: String::from("Chainmail"),  cost:  31, attack: 0, defence: 2},
            Equipment{name: String::from("Splintmail"), cost:  53, attack: 0, defence: 3},
            Equipment{name: String::from("Bandedmail"), cost:  75, attack: 0, defence: 4},
            Equipment{name: String::from("Platemail"),  cost: 102, attack: 0, defence: 5},
        ]
    );

    let rings = Vec::from(
        [
            Equipment{name: String::from("Damage +1"),  cost:  25, attack: 1, defence: 0},
            Equipment{name: String::from("Damage +2"),  cost:  50, attack: 2, defence: 0},
            Equipment{name: String::from("Damage +3"),  cost: 100, attack: 3, defence: 0},
            Equipment{name: String::from("Defence +1"), cost:  20, attack: 0, defence: 1},
            Equipment{name: String::from("Defence +2"), cost:  40, attack: 0, defence: 2},
            Equipment{name: String::from("Defence +3"), cost:  60, attack: 0, defence: 3},
        ]
    );

    let combinations = get_all_equip_combinations(weapons, armour, rings);
    //println!("{combinations:?}");

    let mut successes = Vec::<usize>::new();

    for (weapon, armour, rings, cost) in combinations {

        //println!("\n{:?}, {:?}, {:?}, {}", weapon, armour, rings, cost);
        let mut hero = Character{hit_points: 100, armour_val: 0, attack_val: 0};
        let mut boss = Character{hit_points: 104, armour_val: 1, attack_val: 8};

        let base_weapon_val = weapon.attack;
        let base_armour_val = match &armour {
            Some(armour) => armour.defence,
            None => 0
        };
        let (ring_attack, ring_def) = match &rings {
            Some(rings) => {
                let mut attack_val = 0;
                let mut def_val = 0;
                for ring in rings {
                    attack_val += ring.attack;
                    def_val += ring.defence;
                }
                (attack_val, def_val)
            },
            None => (0, 0)
        };
        let weapon_val = base_weapon_val + ring_attack;
        let armour_val = base_armour_val + ring_def;
        //println!("  Weapon val: {weapon_val}, armour val {armour_val}");
    
        loop {

            let hero_hit= weapon_val;
            let hero_def = armour_val;

            boss.hit_points -= (hero_hit as isize - boss.armour_val as isize).max(1) as isize;

            if boss.hit_points <= 0 {
                // println!("    boss defeated");
                successes.push(cost);

                break;
            }

            hero.hit_points -= (boss.attack_val as isize - hero_def as isize).max(1) as isize;

            if hero.hit_points <= 0 {
                // println!("    Hero defeated failed");
                break;
            }

            // println!("    boss: {} hero: {}", boss.hit_points, hero.hit_points);
        }
    }
    
    let min = successes.iter().min();
    if min.is_some() {
        println!("Lowest cost {}", min.unwrap());
    }

}