# RULES OF THIS WORLD (freeciv ruleset: classic)
Static for the whole trajectory.  Stated once, like the map.

## TECHNOLOGY TREE
technology: prerequisites
  Advanced Flight: Radio + Machine Tools
  Alphabet: (available from the start)
  Amphibious Warfare: Navigation + Tactics
  Astronomy: Mysticism + Mathematics
  Atomic Theory: Theory of Gravity + Physics
  Automobile: Combustion + Steel
  Banking: Trade + The Republic
  Bridge Building: Iron Working + Construction
  Bronze Working: (available from the start)
  Ceremonial Burial: (available from the start)
  Chemistry: University + Medicine
  Chivalry: Feudalism + Horseback Riding
  Code of Laws: Alphabet
  Combined Arms: Mobile Warfare + Advanced Flight
  Combustion: Refining + Explosives
  Communism: Philosophy + Industrialization
  Computers: Mass Production + Miniaturization
  Conscription: Democracy + Metallurgy
  Construction: Masonry + Currency
  Currency: Bronze Working
  Democracy: Banking + Invention
  Economics: Banking + University
  Electricity: Metallurgy + Magnetism
  Electronics: The Corporation + Electricity
  Engineering: The Wheel + Construction
  Environmentalism: Recycling + Space Flight
  Espionage: Communism + Democracy
  Explosives: Gunpowder + Chemistry
  Feudalism: Warrior Code + Monarchy
  Flight: Combustion + Theory of Gravity
  Fusion Power: Nuclear Power + Superconductors
  Genetic Engineering: Medicine + The Corporation
  Guerilla Warfare: Communism + Tactics
  Gunpowder: Invention + Iron Working
  Horseback Riding: (available from the start)
  Industrialization: Railroad + Banking
  Invention: Engineering + Literacy
  Iron Working: Bronze Working + Warrior Code
  Labor Union: Mass Production + Guerilla Warfare
  Laser: Mass Production + Nuclear Power
  Leadership: Chivalry + Gunpowder
  Literacy: Writing + Code of Laws
  Machine Tools: Steel + Tactics
  Magnetism: Iron Working + Physics
  Map Making: Alphabet
  Masonry: (available from the start)
  Mass Production: Automobile + The Corporation
  Mathematics: Alphabet + Masonry
  Medicine: Philosophy + Trade
  Metallurgy: Gunpowder + University
  Miniaturization: Machine Tools + Electronics
  Mobile Warfare: Automobile + Tactics
  Monarchy: Ceremonial Burial + Code of Laws
  Monotheism: Philosophy + Polytheism
  Mysticism: Ceremonial Burial
  Navigation: Seafaring + Astronomy
  Nuclear Fission: Mass Production + Atomic Theory
  Nuclear Power: Nuclear Fission + Electronics
  Philosophy: Mysticism + Literacy
  Physics: Literacy + Navigation
  Plastics: Refining + Space Flight
  Polytheism: Horseback Riding + Ceremonial Burial
  Pottery: (available from the start)
  Radio: Flight + Electricity
  Railroad: Steam Engine + Bridge Building
  Recycling: Mass Production + Democracy
  Refining: Chemistry + The Corporation
  Refrigeration: Sanitation + Electricity
  Robotics: Mobile Warfare + Computers
  Rocketry: Advanced Flight + Electronics
  Sanitation: Engineering + Medicine
  Seafaring: Pottery + Map Making
  Space Flight: Computers + Rocketry
  Stealth: Superconductors + Advanced Flight
  Steam Engine: Physics + Invention
  Steel: Electricity + Industrialization
  Superconductors: Nuclear Power + Laser
  Tactics: Conscription + Leadership
  The Corporation: Economics + Industrialization
  The Republic: Code of Laws + Literacy
  The Wheel: Horseback Riding
  Theology: Feudalism + Monotheism
  Theory of Gravity: Astronomy + University
  Trade: Currency + Code of Laws
  University: Mathematics + Philosophy
  Warrior Code: (available from the start)
  Writing: Alphabet

## UNITS
unit | class | needs tech | cost | attack | defense | hp | firepower | moves | obsoleted by
  Settlers | Land | - | 40 | 0 | 1 | 20 | 1 | 1 | None
  Workers | Land | Pottery | 30 | 0 | 1 | 10 | 1 | 1 | Engineers
  Engineers | Land | Explosives | 40 | 0 | 2 | 20 | 1 | 2 | None
  Warriors | Land | - | 10 | 1 | 1 | 10 | 1 | 1 | Pikemen
  Phalanx | Land | Bronze Working | 20 | 1 | 2 | 10 | 1 | 1 | Pikemen
  Archers | Land | Warrior Code | 30 | 3 | 2 | 10 | 1 | 1 | Musketeers
  Legion | Land | Iron Working | 40 | 4 | 2 | 10 | 1 | 1 | Musketeers
  Pikemen | Land | Feudalism | 20 | 1 | 2 | 10 | 1 | 1 | Musketeers
  Musketeers | Land | Gunpowder | 30 | 3 | 3 | 20 | 1 | 1 | Riflemen
  Partisan | Land | Guerilla Warfare | 50 | 4 | 4 | 20 | 1 | 1 | None
  Alpine Troops | Land | Tactics | 50 | 5 | 5 | 20 | 1 | 1 | None
  Riflemen | Land | Conscription | 40 | 5 | 4 | 20 | 1 | 1 | None
  Marines | Land | Amphibious Warfare | 60 | 8 | 5 | 20 | 1 | 1 | None
  Paratroopers | Land | Combined Arms | 60 | 6 | 4 | 20 | 1 | 1 | None
  Mech. Inf. | Land | Labor Union | 50 | 6 | 6 | 30 | 1 | 3 | None
  Horsemen | Land | Horseback Riding | 20 | 2 | 1 | 10 | 1 | 2 | Knights
  Chariot | Land | The Wheel | 30 | 3 | 1 | 10 | 1 | 2 | Knights
  Knights | Land | Chivalry | 40 | 4 | 2 | 10 | 1 | 2 | Dragoons
  Dragoons | Land | Leadership | 50 | 5 | 2 | 20 | 1 | 2 | Cavalry
  Cavalry | Land | Tactics | 60 | 8 | 3 | 20 | 1 | 2 | Armor
  Armor | Land | Mobile Warfare | 80 | 10 | 5 | 30 | 1 | 3 | None
  Catapult | Land | Mathematics | 40 | 6 | 1 | 10 | 1 | 1 | Cannon
  Cannon | Land | Metallurgy | 40 | 8 | 1 | 20 | 1 | 1 | Artillery
  Artillery | Land | Machine Tools | 50 | 10 | 1 | 20 | 2 | 1 | Howitzer
  Howitzer | Land | Robotics | 70 | 12 | 2 | 30 | 2 | 2 | None
  Fighter | Air | Flight | 60 | 4 | 3 | 20 | 2 | 10 | Stealth Fighter
  Bomber | Air | Advanced Flight | 120 | 12 | 1 | 20 | 2 | 8 | Stealth Bomber
  Helicopter | Helicopter | Combined Arms | 100 | 10 | 3 | 20 | 2 | 6 | None
  Stealth Fighter | Air | Stealth | 80 | 8 | 4 | 20 | 2 | 14 | None
  Stealth Bomber | Air | Stealth | 160 | 18 | 5 | 20 | 2 | 12 | None
  Trireme | Trireme | Map Making | 40 | 1 | 1 | 10 | 1 | 3 | Caravel
  Caravel | Sea | Navigation | 40 | 2 | 1 | 10 | 1 | 3 | Galleon
  Galleon | Sea | Magnetism | 40 | 0 | 2 | 20 | 1 | 4 | Transport
  Frigate | Sea | Magnetism | 50 | 4 | 2 | 20 | 1 | 4 | Ironclad
  Ironclad | Sea | Steam Engine | 60 | 4 | 4 | 30 | 1 | 4 | Destroyer
  Destroyer | Sea | Electricity | 60 | 4 | 4 | 30 | 1 | 6 | None
  Cruiser | Sea | Steel | 80 | 6 | 6 | 30 | 2 | 5 | AEGIS Cruiser
  AEGIS Cruiser | Sea | Rocketry | 100 | 8 | 8 | 30 | 2 | 5 | None
  Battleship | Sea | Automobile | 160 | 12 | 12 | 40 | 2 | 4 | None
  Submarine | Sea | Combustion | 50 | 12 | 2 | 30 | 2 | 5 | None
  Carrier | Sea | Advanced Flight | 160 | 1 | 9 | 40 | 2 | 5 | None
  Transport | Sea | Industrialization | 50 | 0 | 3 | 30 | 1 | 5 | None
  Cruise Missile | Missile | Rocketry | 60 | 18 | 0 | 10 | 3 | 12 | None
  Nuclear | Missile | Rocketry | 160 | 99 | 0 | 10 | 1 | 16 | None
  Diplomat | Land | Writing | 30 | 0 | 0 | 10 | 1 | 2 | Spy
  Spy | Land | Espionage | 30 | 0 | 0 | 10 | 1 | 3 | None
  Caravan | Land | Trade | 50 | 0 | 1 | 10 | 1 | 1 | Freight
  Freight | Land | The Corporation | 50 | 0 | 1 | 10 | 1 | 2 | None
  Explorer | Land | Seafaring | 30 | 0 | 1 | 10 | 1 | 1 | Partisan
  Leader | Land | - | 10 | 0 | 2 | 20 | 1 | 2 | None
  Barbarian Leader | Land | - | 40 | 0 | 0 | 10 | 1 | 2 | None
  AWACS | Air | Advanced Flight | 140 | 0 | 1 | 20 | 1 | 16 | None

## CITY IMPROVEMENTS
improvement | genus | needs tech | cost | upkeep
  Airport | Improvement | Radio | 120 | 3
  Aqueduct | Improvement | Construction | 60 | 2
  Bank | Improvement | Banking | 80 | 2
  Barracks | Improvement | - | 30 | 1
  Barracks II | Improvement | Gunpowder | 30 | 1
  Barracks III | Improvement | Mobile Warfare | 30 | 1
  Cathedral | Improvement | Monotheism | 80 | 3
  City Walls | Improvement | Masonry | 60 | 0
  Coastal Defense | Improvement | Gunpowder | 60 | 1
  Colosseum | Improvement | Construction | 70 | 4
  Courthouse | Improvement | Code of Laws | 60 | 1
  Factory | Improvement | Industrialization | 140 | 4
  Granary | Improvement | Pottery | 40 | 1
  Harbor | Improvement | Seafaring | 40 | 1
  Hydro Plant | Improvement | Electronics | 180 | 4
  Library | Improvement | Writing | 60 | 1
  Marketplace | Improvement | Currency | 60 | 0
  Mass Transit | Improvement | Mass Production | 120 | 4
  Mfg. Plant | Improvement | Robotics | 220 | 6
  Nuclear Plant | Improvement | Nuclear Power | 120 | 2
  Offshore Platform | Improvement | Miniaturization | 120 | 3
  Palace | SmallWonder | Masonry | 70 | 0
  Police Station | Improvement | Communism | 50 | 2
  Port Facility | Improvement | Amphibious Warfare | 60 | 3
  Power Plant | Improvement | Refining | 130 | 4
  Recycling Center | Improvement | Recycling | 140 | 2
  Research Lab | Improvement | Computers | 120 | 3
  SAM Battery | Improvement | Rocketry | 70 | 2
  SDI Defense | Improvement | Laser | 140 | 4
  Sewer System | Improvement | Sanitation | 80 | 2
  Solar Plant | Improvement | Environmentalism | 320 | 4
  Space Component | Special | Plastics | 160 | 0
  Space Module | Special | Superconductors | 320 | 0
  Space Structural | Special | Space Flight | 80 | 0
  Stock Exchange | Improvement | Economics | 120 | 3
  Super Highways | Improvement | Automobile | 120 | 3
  Supermarket | Improvement | Refrigeration | 80 | 3
  Temple | Improvement | Ceremonial Burial | 30 | 1
  University | Improvement | University | 120 | 3
  Apollo Program | GreatWonder | Space Flight | 600 | 0
  A.Smith's Trading Co. | GreatWonder | Economics | 400 | 0
  Colossus | GreatWonder | Bronze Working | 100 | 0
  Copernicus' Observatory | GreatWonder | Astronomy | 200 | 0
  Cure For Cancer | GreatWonder | Genetic Engineering | 600 | 0
  Darwin's Voyage | GreatWonder | Railroad | 300 | 0
  Eiffel Tower | GreatWonder | Steam Engine | 100 | 0
  Great Library | GreatWonder | Literacy | 300 | 0
  Great Wall | GreatWonder | Masonry | 300 | 0
  Hanging Gardens | GreatWonder | Pottery | 200 | 0
  Hoover Dam | GreatWonder | Electronics | 600 | 0
  Isaac Newton's College | GreatWonder | Theory of Gravity | 300 | 0
  J.S. Bach's Cathedral | GreatWonder | Theology | 400 | 0
  King Richard's Crusade | GreatWonder | Engineering | 150 | 0
  Leonardo's Workshop | GreatWonder | Invention | 400 | 0
  Lighthouse | GreatWonder | Map Making | 200 | 0
  Magellan's Expedition | GreatWonder | Navigation | 400 | 0
  Manhattan Project | GreatWonder | Nuclear Fission | 600 | 0
  Marco Polo's Embassy | GreatWonder | Trade | 200 | 0
  Michelangelo's Chapel | GreatWonder | Monotheism | 400 | 0
  Oracle | GreatWonder | Mysticism | 100 | 0
  Pyramids | GreatWonder | Masonry | 200 | 0
  SETI Program | GreatWonder | Computers | 600 | 0
  Shakespeare's Theater | GreatWonder | Medicine | 200 | 0
  Statue of Liberty | GreatWonder | Democracy | 400 | 0
  Sun Tzu's War Academy | GreatWonder | Feudalism | 300 | 0
  United Nations | GreatWonder | Communism | 600 | 0
  Women's Suffrage | GreatWonder | Industrialization | 600 | 0
  Coinage | Special | - | 999 | 0

## TERRAIN
terrain | move cost | defense bonus % | food/shield/trade | mining | irrigation
  Lake | 1 | 0 | 1/0/2 | +0 shield | +0 food
  Ocean | 1 | 0 | 1/0/2 | +0 shield | +0 food
  Deep Ocean | 1 | 0 | 1/0/2 | +0 shield | +0 food
  Glacier | 2 | 0 | 0/0/0 | +1 shield | +0 food
  Desert | 1 | 0 | 0/1/0 | +1 shield | +1 food
  Forest | 2 | 50 | 1/2/0 | +0 shield | +0 food
  Grassland | 1 | 0 | 2/0/0 | +0 shield | +1 food
  Hills | 2 | 100 | 1/0/0 | +3 shield | +1 food
  Jungle | 2 | 50 | 1/0/0 | +0 shield | +0 food
  Mountains | 3 | 200 | 0/1/0 | +1 shield | +0 food
  Plains | 1 | 0 | 1/1/0 | +0 shield | +1 food
  Swamp | 2 | 50 | 1/0/0 | +0 shield | +0 food
  Tundra | 1 | 0 | 1/0/0 | +0 shield | +1 food

## GOVERNMENTS (classic)
(helptext verbatim from governments.ruleset)
  Anarchy: Anarchy is simply the absence of any recognizable government. Citizens are disorganized and unproductive, and will spend all income as quickly as possible, rather than paying taxes or conducting research. Anarchy offers slightly less corruption than Despotism, but slightly more unhappiness.
  Despotism: Under Despotism, you are the absolute ruler of your people. Your control over your citizens is maintained largely by martial law. Despotism suffers the highest level of corruption of all forms of government.
  Monarchy: Under Monarchy, a king or queen serves as a hereditary figurehead for your government. Monarchy suffers the same small amount of corruption that the Republic does.
  Communism: A Communist government is based on the ideal that all people are equal. All goods are owned by the state, rather than by private citizens. Communism gives a balance between military and commercial styles of government. Under Communism, corruption does not vary by distance from the capital; all cities (including the capital) have a modest amount of corruption.
  Republic: Under a Republican government, citizens hold an election to select a representative who will govern them; since elected leaders must remain popular to remain in control, citizens are given a greater degree of freedom. Citizens under the Republic become unhappy easily, but the self-sufficiency of your citizens allows high levels of trade.
  Democracy: Under Democracy, citizens govern directly by voting on issues. Democracy offers the highest possible level of trade, but also offers the most potential for unhappiness. There is no corruption during Democracy, but citizens become very upset during wars.

Effects conditioned on a government (verbatim, not paraphrased):
  Upkeep_Free = 99   when Gov=Anarchy
  Tech_Upkeep_Free = 9999   when Gov=Anarchy
  Make_Content_Mil = 1   when Gov=Republic
  Output_Waste = 25   when Gov=Anarchy; OutputType=Trade
  Output_Waste_By_Distance = 200   when Gov=Anarchy; OutputType=Trade
  Output_Waste = 37   when Gov=Despotism; OutputType=Trade
  Output_Waste_By_Distance = 400   when Gov=Despotism; OutputType=Trade
  Output_Waste = 15   when Gov=Monarchy; OutputType=Trade
  Output_Waste_By_Distance = 200   when Gov=Monarchy; OutputType=Trade
  Output_Waste = 20   when Gov=Communism; OutputType=Trade
  Output_Waste = 15   when Gov=Republic; OutputType=Trade
  Output_Waste_By_Distance = 200   when Gov=Republic; OutputType=Trade
  Upkeep_Factor = 1   when Gov=Republic; OutputType=Food
  Upkeep_Factor = 1   when Gov=Democracy; OutputType=Food
  Unhappy_Factor = 1   when Gov=Republic
  Unhappy_Factor = 2   when Gov=Democracy
  Unit_Upkeep_Free_Per_City = 3   when Gov=Anarchy; OutputType=Shield
  Unit_Upkeep_Free_Per_City = 3   when Gov=Despotism; OutputType=Shield
  Unit_Upkeep_Free_Per_City = 3   when Gov=Monarchy; OutputType=Shield
  Unit_Upkeep_Free_Per_City = 3   when Gov=Communism; OutputType=Shield
  Civil_War_Chance = 90   when Gov=Anarchy
  Civil_War_Chance = 80   when Gov=Despotism
  Civil_War_Chance = 70   when Gov=Monarchy
  Civil_War_Chance = 50   when Gov=Communism
  Civil_War_Chance = 40   when Gov=Republic
  Civil_War_Chance = 30   when Gov=Democracy
  Empire_Size_Base = 9   when Gov=Anarchy
  Empire_Size_Base = 10   when Gov=Despotism
  Empire_Size_Base = 11   when Gov=Monarchy
  Empire_Size_Base = 12   when Gov=Communism
  Empire_Size_Base = 13   when Gov=Republic
  Empire_Size_Base = 14   when Gov=Democracy
  Empire_Size_Step = 6   when Gov=Anarchy
  Empire_Size_Step = 10   when Gov=Despotism
  Empire_Size_Step = 12   when Gov=Monarchy
  Empire_Size_Step = 14   when Gov=Republic
  Empire_Size_Step = 16   when Gov=Democracy
  Max_Rates = 100   when Gov=Anarchy
  Max_Rates = 60   when Gov=Despotism
  Max_Rates = 70   when Gov=Monarchy
  Max_Rates = 80   when Gov=Communism
  Max_Rates = 80   when Gov=Republic
  Max_Rates = 100   when Gov=Democracy
  Martial_Law_Each = 1   when Gov=Anarchy
  Martial_Law_Each = 1   when Gov=Despotism
  Martial_Law_Each = 1   when Gov=Monarchy
  Martial_Law_Each = 2   when Gov=Communism
  Martial_Law_Max = 100   when Gov=Anarchy
  Martial_Law_Max = 100   when Gov=Despotism
  Martial_Law_Max = 3   when Gov=Monarchy
  Martial_Law_Max = 3   when Gov=Communism
  Rapture_Grow = 3   when Gov=Republic
  Rapture_Grow = 3   when Gov=Democracy
  Veteran_Build = 1   when Gov=Communism; UnitFlag=Diplomat
  Output_Penalty_Tile = 2   when Gov=Anarchy
  Output_Penalty_Tile = 2   when Gov=Despotism
  Output_Inc_Tile_Celebrate = 1   when OutputType=Trade; Gov=Monarchy
  Output_Inc_Tile_Celebrate = 1   when OutputType=Trade; Gov=Communism
  Output_Inc_Tile = 1   when Gov=Republic; OutputType=Trade
  Output_Inc_Tile = 1   when Gov=Democracy; OutputType=Trade
  Revolution_Unhappiness = 2   when Gov=Democracy
  Has_Senate = 1   when Gov=Republic
  Has_Senate = 1   when Gov=Democracy
  Inspire_Partisans = 1   when Gov=Communism; Tech=Communism; Tech=Gunpowder; Tech=Guerilla Warfare; NationGroup=Barbarian (absent)
  Inspire_Partisans = 1   when Gov=Democracy; Tech=Communism; Tech=Gunpowder; Tech=Guerilla Warfare; NationGroup=Barbarian (absent)
  Make_Content = 1   when Gov=Democracy; Building=Courthouse
  Output_Bonus = 75   when Gov=Despotism; Building=Palace; OutputType=Shield
  Output_Bonus = 50   when Gov=Monarchy; Building=Palace; OutputType=Shield
  Make_Content_Mil = 1   when Gov=Republic; Building=Police Station; Building=Women's Suffrage (absent)
  Make_Content_Mil = 2   when Gov=Democracy; Building=Police Station; Building=Women's Suffrage (absent)
  Make_Content_Mil = 1   when Gov=Republic; Building=Women's Suffrage
  Make_Content_Mil = 2   when Gov=Democracy; Building=Women's Suffrage

(counts: 87 technologies, 52 units, 68 improvements, 13 terrains)