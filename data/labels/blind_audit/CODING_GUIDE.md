# Blind hazard-coding guide

You are coding San Francisco Department of Emergency Management (DEM) dispatch records of
autonomous-vehicle (AV, robotaxi) **blocking incidents**: events where a driverless vehicle
became immobilized and obstructed the roadway. Each row of `coding_sheet.csv` is one
incident. Assign each incident **exactly one** hazard code from the taxonomy below.

## What a "hazard" means here

A hazard is defined by its **consequence**: what the immobilized AV obstructed or put at
risk. It is *not* the cause. A dead battery, a dark traffic signal, a software fault, or a
passenger who left the door open are causes; code the incident by what the stopped AV
affected.

## Hazard classes

| Code | Hazard | Definition |
|---|---|---|
| H1 | AV-involved collision | The AV made (or was reported to make) contact with another road user, vehicle, or animal, leaving the AV or the scene blocking the roadway. |
| H2 | Emergency-response obstruction | The AV obstructed an emergency response (medics, fire, police) or became stuck inside an active emergency scene (fire, crash scene, police investigation). |
| H3 | Transit / rail obstruction | The AV blocked a public-transit vehicle, lane, or track (Muni bus or rail, cable car) or a railroad track. |
| H4 | Pedestrian / accessibility obstruction | The AV blocked a crosswalk, curb ramp, or bike lane, pushing vulnerable road users into traffic. |
| H5 | Occupant-related immobilization | A passenger condition or action immobilized the AV: unresponsive/asleep rider, occupant trapped or calling for help, or a door left open. |
| H6 | Multi-AV clustering / gridlock | Two or more AVs immobilized together at one location, with none of the H1-H5 consequences. |
| H7 | Single-AV traffic obstruction | One AV stalled in a travel lane, intersection, or roadway, obstructing general traffic only. Default class. |

## Rules

1. **Precedence.** If an incident fits more than one class, choose the most severe: the
   **lowest-numbered** code that applies (H1 before H2 before ... before H7).
2. **H6 uses the AV count.** Use the `n_avs` column for the number of AVs. H6 applies when
   `n_avs` is 2 or more and none of H1-H5 applies. If `n_avs` is `unknown`, treat it as 1.
3. **Code only what the text supports.** Do not infer a consequence the narrative does not
   state. Many narratives are truncated mid-sentence; code the visible text.
4. **Default.** If nothing points to H1-H6, code H7. A missing narrative is coded H7.
5. **Ambiguity.** When two codes are both defensible, pick one, set confidence to `low`,
   and say why in the rationale.

## Dispatcher shorthand

| Abbreviation | Meaning |
|---|---|
| blkng / blking | blocking |
| ntfyd / advd | notified / advised |
| LP | license plate |
| RP / Rp | reporting party (the caller) |
| PD / SFPD / CHP | police / SF police / California Highway Patrol |
| SFFD | SF Fire Department |
| Muni | SF public transit (buses, light rail, cable cars) |
| ifo | in front of |
| OS | on scene |
| GOA / UTL | gone on arrival / unable to locate |
| LL / cb / tx | landline / callback / text message |
| iao | in area of |
| veh / vehs | vehicle(s) |
| Wht Jag / I-pace | white Jaguar I-Pace (the vehicle model Waymo uses) |
