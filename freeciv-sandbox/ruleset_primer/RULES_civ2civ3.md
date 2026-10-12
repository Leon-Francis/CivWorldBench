# RULES OF THIS WORLD (freeciv ruleset: civ2civ3)
Static for the whole trajectory.  Stated once, like the map.

## TECHNOLOGY TREE
technology: prerequisites
  Advanced Flight: Radio + Machine Tools
  Alphabet: (available from the start)
  Amphibious Warfare: Engineering + Tactics
  Astronomy: Mysticism + Mathematics
  Atomic Theory: Chemistry + Refrigeration
  Automobile: Combustion + Steel
  Banking: Trade + The Republic
  Bridge Building: The Wheel + Construction
  Bronze Working: (available from the start)
  Ceremonial Burial: (available from the start)
  Chemistry: University + Medicine
  Chivalry: Feudalism + Horseback Riding
  Code of Laws: Alphabet
  Combined Arms: Mobile Warfare + Advanced Flight
  Combustion: Refining + Engineering
  Communism: Theology + Industrialization
  Computers: Miniaturization + Radio
  Conscription: Democracy + Metallurgy
  Construction: Masonry + Iron Working
  Currency: Bronze Working
  Democracy: Banking + Invention
  Economics: Banking + University
  Electricity: Metallurgy + Theory of Gravity
  Electronics: The Corporation + Electricity
  Engineering: Electricity + Steam Engine
  Environmentalism: Recycling + Space Flight
  Espionage: Communism + Democracy
  Explosives: Gunpowder + Chemistry
  Feudalism: Iron Working + Monarchy
  Flight: Combustion + Tactics
  Fusion Power: Superconductors + Labor Union
  Genetic Engineering: Refrigeration + The Corporation
  Guerilla Warfare: Communism + Tactics
  Gunpowder: Invention + Feudalism
  Horseback Riding: (available from the start)
  Industrialization: Railroad + Banking
  Invention: Bridge Building + Literacy
  Iron Working: Bronze Working + Warrior Code
  Labor Union: Communism + Mobile Warfare
  Laser: Computers + Nuclear Power
  Leadership: Chivalry + Gunpowder
  Literacy: Writing + Code of Laws
  Machine Tools: Steel + Tactics
  Magnetism: Seafaring + Astronomy
  Map Making: Alphabet
  Masonry: (available from the start)
  Mass Production: Automobile + The Corporation
  Mathematics: Alphabet + Masonry
  Medicine: Philosophy + Trade
  Metallurgy: Gunpowder + Mathematics
  Miniaturization: Combustion + Electronics
  Mobile Warfare: Automobile + Machine Tools
  Monarchy: Ceremonial Burial + Code of Laws
  Monotheism: Astronomy + Polytheism
  Mysticism: Ceremonial Burial
  Navigation: Invention + Physics
  Nuclear Fission: Mass Production + Atomic Theory
  Nuclear Power: Nuclear Fission + Miniaturization
  Philosophy: Mysticism + Literacy
  Physics: The Wheel + Magnetism
  Plastics: Mass Production + Robotics
  Polytheism: Horseback Riding + Ceremonial Burial
  Pottery: (available from the start)
  Radio: Flight + Electronics
  Railroad: Steam Engine + Metallurgy
  Recycling: Mass Production + Sanitation
  Refining: Explosives + Industrialization
  Refrigeration: Sanitation + Electricity
  Robotics: Mobile Warfare + Computers
  Rocketry: Advanced Flight + Miniaturization
  Sanitation: Bridge Building + Medicine
  Seafaring: Pottery + Map Making
  Space Flight: Computers + Rocketry
  Stealth: Plastics + Space Flight
  Steam Engine: Chemistry + Navigation
  Steel: Engineering + Industrialization
  Superconductors: Space Flight + Laser
  Tactics: Conscription + Leadership
  The Corporation: Economics + Industrialization
  The Republic: Code of Laws + Literacy
  The Wheel: Horseback Riding
  Theology: Philosophy + Monotheism
  Theory of Gravity: Physics + University
  Trade: Currency + Pottery
  University: Mathematics + Philosophy
  Warrior Code: (available from the start)
  Writing: Alphabet

## UNITS
unit | class | needs tech | cost | attack | defense | hp | firepower | moves | obsoleted by
  Settlers | Small Land | - | 30 | 0 | 1 | 20 | 1 | 1 | None
  Migrants | Small Land | Pottery | 10 | 0 | 1 | 20 | 1 | 1 | None
  Workers | Small Land | - | 20 | 0 | 1 | 10 | 1 | 1 | Engineers
  Engineers | Small Land | Explosives | 30 | 0 | 2 | 20 | 1 | 2 | None
  Warriors | Land | - | 10 | 1 | 1 | 10 | 1 | 1 | Musketeers
  Phalanx | Land | Bronze Working | 20 | 1 | 2 | 10 | 1 | 1 | Pikemen
  Archers | Land | Warrior Code | 20 | 3 | 1 | 10 | 1 | 1 | Musketeers
  Legion | Land | Iron Working | 30 | 4 | 2 | 10 | 1 | 1 | Musketeers
  Pikemen | Land | Feudalism | 30 | 2 | 3 | 10 | 1 | 1 | Musketeers
  Musketeers | Land | Gunpowder | 40 | 3 | 3 | 20 | 1 | 1 | Riflemen
  Riflemen | Land | Conscription | 50 | 5 | 4 | 20 | 1 | 1 | Marines
  Alpine Troops | Land | Tactics | 60 | 7 | 4 | 20 | 1 | 1 | None
  Partisan | Land | Guerilla Warfare | 60 | 4 | 5 | 20 | 1 | 1 | None
  Fanatics | Land | Guerilla Warfare | 20 | 5 | 5 | 20 | 1 | 1 | None
  Marines | Land | Amphibious Warfare | 60 | 8 | 5 | 20 | 1 | 1 | None
  Paratroopers | Land | Combined Arms | 60 | 6 | 4 | 20 | 1 | 1 | None
  Mech. Inf. | Big Land | Labor Union | 70 | 6 | 6 | 30 | 1 | 3 | None
  Horsemen | Land | Horseback Riding | 20 | 2 | 1 | 10 | 1 | 2 | Knights
  Chariot | Big Land | The Wheel | 30 | 4 | 1 | 10 | 1 | 2 | Dragoons
  Elephants | Land | Polytheism | 30 | 3 | 2 | 10 | 1 | 2 | Dragoons
  Crusaders | Land | Monotheism | 40 | 5 | 1 | 10 | 1 | 2 | Dragoons
  Knights | Land | Chivalry | 40 | 4 | 3 | 10 | 1 | 2 | Dragoons
  Dragoons | Land | Leadership | 50 | 5 | 2 | 20 | 1 | 2 | Cavalry
  Cavalry | Land | Tactics | 60 | 8 | 3 | 20 | 1 | 2 | Armor
  Armor | Big Land | Mobile Warfare | 90 | 10 | 5 | 30 | 1 | 3 | None
  Catapult | Big Land | Mathematics | 40 | 6 | 1 | 10 | 1 | 1 | Cannon
  Cannon | Big Land | Metallurgy | 50 | 8 | 1 | 20 | 1 | 1 | Artillery
  Artillery | Big Land | Machine Tools | 60 | 10 | 2 | 20 | 1 | 1 | Howitzer
  Howitzer | Big Land | Robotics | 80 | 12 | 2 | 30 | 1 | 1 | None
  Fighter | Air | Flight | 60 | 4 | 4 | 20 | 1 | 10 | Stealth Fighter
  Bomber | Air | Advanced Flight | 100 | 6 | 2 | 30 | 1 | 8 | Stealth Bomber
  Helicopter | Helicopter | Combined Arms | 70 | 5 | 3 | 30 | 1 | 6 | None
  Stealth Fighter | Air | Stealth | 80 | 8 | 8 | 20 | 1 | 14 | None
  Stealth Bomber | Air | Stealth | 120 | 9 | 5 | 30 | 1 | 12 | None
  Trireme | Trireme | Map Making | 20 | 0 | 1 | 10 | 1 | 3 | Transport
  Caravel | Sea | Magnetism | 30 | 0 | 2 | 10 | 1 | 3 | Galleon
  Galleon | Sea | Navigation | 40 | 0 | 2 | 20 | 1 | 4 | Transport
  Frigate | Sea | Navigation | 40 | 3 | 2 | 20 | 1 | 4 | Ironclad
  Ironclad | Sea | Steam Engine | 50 | 4 | 3 | 20 | 1 | 4 | Destroyer
  Destroyer | Sea | Engineering | 60 | 4 | 4 | 30 | 1 | 6 | None
  Cruiser | Sea | Steel | 80 | 6 | 6 | 30 | 1 | 5 | AEGIS Cruiser
  AEGIS Cruiser | Sea | Rocketry | 100 | 8 | 8 | 30 | 1 | 5 | None
  Battleship | Sea | Mass Production | 160 | 12 | 12 | 30 | 1 | 4 | None
  Submarine | Sea | Combustion | 70 | 12 | 5 | 20 | 1 | 5 | None
  Carrier | Sea | Advanced Flight | 150 | 0 | 9 | 40 | 1 | 5 | None
  Transport | Sea | Engineering | 50 | 0 | 3 | 30 | 1 | 5 | None
  Cruise Missile | Missile | Rocketry | 50 | 18 | 0 | 10 | 1 | 16 | None
  Nuclear | Missile | Nuclear Fission | 160 | 99 | 0 | 10 | 1 | 0 | None
  Diplomat | Small Land | Alphabet | 30 | 0 | 0 | 10 | 1 | 2 | Spy
  Spy | Small Land | Espionage | 30 | 0 | 0 | 10 | 1 | 3 | None
  Caravan | Merchant | Trade | 50 | 0 | 1 | 10 | 1 | 1 | Freight
  Freight | Merchant | The Corporation | 50 | 0 | 1 | 10 | 1 | 2 | None
  Explorer | Small Land | Seafaring | 30 | 0 | 1 | 10 | 1 | 1 | None
  Leader | Land | - | 10 | 0 | 2 | 20 | 1 | 2 | None
  Barbarian Leader | Land | - | 40 | 0 | 0 | 10 | 1 | 2 | None
  AWACS | Air | Advanced Flight | 140 | 0 | 1 | 20 | 1 | 16 | None

## CITY IMPROVEMENTS
improvement | genus | needs tech | cost | upkeep
  Airport | Improvement | Radio | 120 | 3
  Aqueduct | Improvement | Construction | 60 | 1
  Aqueduct, Lake | Improvement | - | 20 | 0
  Aqueduct, River | Improvement | - | 20 | 0
  Bank | Improvement | Banking | 80 | 1
  Barracks | Improvement | - | 30 | 1
  Barracks II | Improvement | Gunpowder | 30 | 1
  Barracks III | Improvement | Mobile Warfare | 30 | 1
  Cathedral | Improvement | Monotheism | 80 | 3
  City Walls | Improvement | Masonry | 30 | 0
  Coastal Defense | Improvement | Navigation | 60 | 1
  Amphitheater | Improvement | Construction | 70 | 3
  Courthouse | Improvement | Code of Laws | 60 | 1
  Factory | Improvement | Industrialization | 140 | 3
  Granary | Improvement | Pottery | 40 | 1
  Harbor | Improvement | Seafaring | 40 | 1
  Hydro Plant | Improvement | Electronics | 180 | 4
  Library | Improvement | Writing | 60 | 1
  Marketplace | Improvement | Currency | 60 | 0
  Mass Transit | Improvement | Mass Production | 120 | 2
  Mfg. Plant | Improvement | Plastics | 220 | 4
  Nuclear Plant | Improvement | Nuclear Power | 240 | 2
  Offshore Platform | Improvement | Miniaturization | 120 | 3
  Palace | SmallWonder | - | 70 | 0
  Ecclesiastical Palace | SmallWonder | Theology | 140 | 0
  Police Station | Improvement | Communism | 50 | 2
  Port Facility | Improvement | Amphibious Warfare | 60 | 2
  Power Plant | Improvement | Refining | 130 | 4
  Recycling Center | Improvement | Recycling | 140 | 2
  Research Lab | Improvement | Computers | 120 | 3
  SAM Battery | Improvement | Rocketry | 70 | 2
  SDI Defense | Improvement | Laser | 140 | 3
  Sewer System | Improvement | Sanitation | 80 | 2
  Solar Plant | Improvement | Environmentalism | 320 | 4
  Space Component | Special | Superconductors | 300 | 0
  Space Module | Special | Environmentalism | 600 | 0
  Space Structural | Special | Space Flight | 150 | 0
  Stock Exchange | Improvement | Economics | 120 | 2
  Super Highways | Improvement | Automobile | 120 | 3
  Supermarket | Improvement | Refrigeration | 80 | 3
  Temple | Improvement | Ceremonial Burial | 30 | 1
  University | Improvement | University | 120 | 2
  Apollo Program | GreatWonder | Space Flight | 600 | 0
  A.Smith's Trading Co. | GreatWonder | Economics | 400 | 0
  Colossus | GreatWonder | Bronze Working | 200 | 0
  Copernicus' Observatory | GreatWonder | Astronomy | 300 | 0
  Cure For Cancer | GreatWonder | Genetic Engineering | 600 | 0
  Darwin's Voyage | GreatWonder | Steam Engine | 300 | 0
  Eiffel Tower | GreatWonder | Railroad | 300 | 0
  Great Library | GreatWonder | Literacy | 300 | 0
  Great Wall | GreatWonder | Construction | 300 | 0
  Hanging Gardens | GreatWonder | Pottery | 200 | 0
  Hoover Dam | GreatWonder | Electronics | 600 | 0
  Isaac Newton's College | GreatWonder | Theory of Gravity | 400 | 0
  J.S. Bach's Cathedral | GreatWonder | Theology | 400 | 0
  King Richard's Crusade | GreatWonder | Chivalry | 300 | 0
  Leonardo's Workshop | GreatWonder | Invention | 400 | 0
  Lighthouse | GreatWonder | Map Making | 200 | 0
  Magellan's Expedition | GreatWonder | Magnetism | 400 | 0
  Manhattan Project | GreatWonder | Nuclear Fission | 600 | 0
  Marco Polo's Embassy | GreatWonder | Trade | 200 | 0
  Michelangelo's Chapel | GreatWonder | Monotheism | 400 | 0
  Mausoleum of Mausolos | GreatWonder | Ceremonial Burial | 200 | 0
  Statue of Zeus | GreatWonder | Polytheism | 200 | 0
  Temple of Artemis | GreatWonder | Mysticism | 200 | 0
  Pyramids | GreatWonder | Masonry | 200 | 0
  Internet | GreatWonder | Computers | 600 | 0
  Shakespeare's Theater | GreatWonder | Sanitation | 300 | 0
  Statue of Liberty | GreatWonder | Democracy | 400 | 0
  Sun Tzu's War Academy | GreatWonder | Feudalism | 300 | 0
  United Nations | GreatWonder | Communism | 600 | 0
  Women's Suffrage | GreatWonder | Conscription | 600 | 0
  Coinage | Special | - | 999 | 0

## TERRAIN
terrain | move cost | defense bonus % | food/shield/trade | mining | irrigation
  Lake | 1 | 0 | 2/0/2 | +0 shield | +0 food
  Ocean | 1 | 0 | 1/0/2 | +0 shield | +0 food
  Deep Ocean | 1 | 0 | 1/0/2 | +0 shield | +0 food
  Glacier | 2 | 0 | 0/0/0 | +1 shield | +0 food
  Desert | 1 | 0 | 0/1/0 | +1 shield | +1 food
  Forest | 2 | 25 | 1/2/0 | +0 shield | +0 food
  Grassland | 1 | 0 | 2/0/0 | +0 shield | +1 food
  Hills | 2 | 50 | 1/1/0 | +2 shield | +1 food
  Jungle | 2 | 25 | 1/1/0 | +0 shield | +0 food
  Mountains | 3 | 100 | 0/1/0 | +2 shield | +0 food
  Plains | 1 | 0 | 1/1/0 | +0 shield | +1 food
  Swamp | 2 | 25 | 1/0/0 | +0 shield | +1 food
  Tundra | 1 | 0 | 1/0/0 | +1 shield | +0 food

## GOVERNMENTS (civ2civ3)
(verbatim from README.civ2civ3 -- the ruleset authors' own table)
------------
                  Anarch   Tribal   Despot   Monarc   Fundam   Republ   Democr   Federat  Commun
Tax/Lux/Sci Rate  0        60       60       70       70       80       90       90       80
Output per tile # -1(if>2) -1(if>2) -1(if>2) 0        0        +1Trade  +1Trade  0        0
Celebration Bonus (no -1)  (no -1)  (no -1) (+1Trade) 0       (+1Trade)(+1Trade)(+1Trade) 0

Unit Upkeep       Shield   Shield   Coin     Coin     Coin(x2) Shield   Coin(x2) Coin(x2) Shield
Free Units        2        2        2        3        2        1        1        2        3
War Unhappiness   0        0        0       -1       -1       -1       -2       -1       -1
Martial Law      +1        0       +1       +1        0        0        0        0       +2

Civil war %       99%      90%      80%      70%      60%      40%      30%      20%      50%
Empire Size       0        /12      /10      /14      /20      /16      /32      /24      /28
Special          +NoUpkeep +Veteran +75%Gold +50%Gold +Tithes  -Revolt -Revolt   +Luxury  +NoUpkeep
                 -50%Lux  (Land/Mil)(Palace) (Palace) -50%Sci          -Senate   -Senate  (Cost=1)
                 -NoTaxes +Content                    +Fanatic         +Partisan          +Partisan

Base Corruption   0%       30%      20%      10%      15%      25%      5%       0%       30%
Base Waste        30%      0%       10%      20%      15%      5%       25%      30%      0
By Distance *    +1%      +1%      +1%      +1%      +1%      +1%      +1%      +0%      +0%

* Corruption by distance is doubled (+2% per tile of distance to
  capital, or +1% for Federation) until The Corporation is researched by
  the player.
  Waste by distance is doubled (+2% per tile of distance to capital)
  until Trade is researched by the player.
# The output per tile penalty is negated by The Pyramids wonder, or
  when Railroad is researched by the player.
+ Partisans appear around conquered cities (democratic or communist) if
  Guerilla Warfare has been researched by any player.

- Added Fundamentalism (available with Theology), Tribal (available at
  start) and Federation (available with Economics).

- "Base Corruption/Waste": Modern governments cause less Corruption in
  Trade, but more Waste in Production. The total % lost (Waste +
  Corruption) is equal for all governments.

- "Empire Size" (number of cities before they start causing extra
  unhappiness): Modern governments have double size limit than their
  ancient counterpart. Empire Base Size is always equal to Empire Step
  Size.

- "Unit Upkeep": Tribal, Republic and Communism use the standard unit
  upkeep in shields, while the other governments use upkeep in coins (x2
  coins per unit for modern governments).

- "Free Units": Means units free of shield/coin upkeep and also free of
  unhappiness due to military activity. By default 2 free units per city.

- "War Unhappiness": Most governments cause unhappiness when military
  units end the turn outside national borders (same as Republic with civ2
  rules).

- "Martial Law": Max 3 units can apply martial law (was max 20 for
  Despotism and Anarchy).

- The bonus "+1 Trade" (for every worked tile that is already producing
  trade) under Republic and Democracy does not affect oceanic tiles.

- Republic gets "Revolution when unhappy" effect. Federation gets
  "Has senate" effect. Democracy gets both effects.

- Spies and Diplomats built under Communism or Federation will start at
  the first veteran level. Federation gets +2 Luxury points in every city.

- Under Communism you do not pay upkeep for buildings that cost 1 coin.
  Communism cancels the bonus from Mysticism and Theology (to Temples and
  Cathedrals).

- Under Fundamentalism, your cities receive extra Gold from tithes
  (from buildings that make citizens content), but -50% to Science (like
  subtracting the effect of a library in each city).

- The output bonus of Palace (under Despotism and Monarchy) switched
  from Production to Gold.

- Tribal government increases by half the chance of ground units
  getting the next veteran level after a battle. Every city gets 1 extra
  content citizen, but units can't apply martial law.

TIP: There is some hurry to get Monarchy or Republic (or to build
     Pyramids), in order to avoid the initial penalty to output per tile.
     However, the Corruption and Waste are evened for all governments, so
     there are no big differences, and each government could be useful in
     different situations.



(counts: 87 technologies, 56 units, 73 improvements, 13 terrains)