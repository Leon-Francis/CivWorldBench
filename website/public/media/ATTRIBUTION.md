# Map artwork and evidence provenance

The map bitmaps are derived from the recorded Freeciv 3.1.6 experiment saves.
Terrain, roads, city locations, sizes, walls and units follow those saved states.
All cities use a uniform European sprite style for this presentation; this
does not assert a civilization's original city style. Maps show full-information
states, not the player-scoped observation supplied to the decision agent.
Case images show one execution per None, Flash and Kimi condition from T60 through T210.
Curves show those individual trajectories; the separate study aggregates cover all 76 worlds.
Unit type, owner, position, hit points and transport relation are read directly
from each save. Ruleset graphic tags select the original Amplio unit sprites
loaded by Hexemplio. One sprite represents each owner/tile stack; stack badges
count all units, including cargo. Transported units are not drawn independently.
For stacks, the highest ruleset defense value selects the representative sprite,
then attack value and unit ID break ties; no units are added by this selection.
Colored bases and flags identify civilizations, not diplomatic relationships.
Per-turn movement marks saved endpoints, not a recovered route. Appearance and
disappearance do not establish creation or death. Same-ID type and owner changes
are recorded separately. Map wrapping can repeat an edge of the same world.
Territory overlays use the saved per-tile owner grid, including owned water;
unclaimed tiles remain untinted. The polygons follow Hexemplio's 126 by 64 pixel
isometric hex geometry with a 16 pixel hex side. Adjacent same-owner tile edges
are merged into region contours; borders are not inferred from city radii.
City label colors follow each city's current owner. Territory and city totals
cover the complete recorded world, including portions outside the viewport.

## Freeciv Hexemplio artwork

Hexemplio artwork is distributed under GPL-2.0-or-later with Freeciv.
Original sprite artists include Peter Arbor, GriffonSpade, Tim F. Smith,
Andreas Rosdal, Daniel Speyer, Architetto Francesco, Vegard Stolpnessaeter,
Canik, CaptainTVK, Erwan, Smiley, Hogne Haskjold, and the credited artists
of Battle for Wesnoth, FreeCol, and OpenGameArt. See the upstream spec files
for individual sprite credits and the original spellings.

Upstream source: https://github.com/freeciv/freeciv/tree/R3_1_6/data/hexemplio
License: https://github.com/freeciv/freeciv/blob/R3_1_6/COPYING
Unit artwork: https://github.com/freeciv/freeciv/tree/R3_1_6/data/amplio2
Unit artists include Alex Mor, Allard H. S. Hofelt, Bebro, Captain Nemo,
CapTVK, Curt Sibling, Erwan, Fairline, GoPostal, Oprisan Sorin, Tanelorn,
Paul Klein Lankhorst / GukGuk, Andrew Livings, Vodvakov, J. W. Bjerk / Eleazar,
qwm, FiftyNine, XYZ, GriffonSpade, VladimirSlavik, ngunjaca and omero.
The derived map and overview bitmaps retain this artwork license.
Displayed scientific summaries are CC-BY-4.0. The benchmark dataset is in preparation.

The adjacent provenance file records the upstream artwork source hashes.
The page includes only the summaries and display coordinates needed for its figures
and two selected examples. Benchmark records and game saves are not distributed.

Full-game case evidence lives in each case's `frames.json`. Territory states are
shared when the saved owner grid is identical. Unit-change records concern units
owned at either endpoint; global unit inventories remain represented by authentic
sprites and per-owner counts. An eliminated player's later frames remain actual
world saves, including any subsequent score changes. No scores are carried forward.
