# Bending the Arc: Curvilinear Acceleration in the Combine Hoop Drill as a Determinant of NFL Pass Rush Get-Off and Quick Pressure Efficiency

**Track:** Open Track  
**Author:** Abenezer ([@abenih](https://www.kaggle.com/abenih))  
**Competition:** NFL Big Data Bowl 2027  
**Attached Public Notebook:** [https://www.kaggle.com/code/abenih/notebook09af53edae](https://www.kaggle.com/code/abenih/notebook09af53edae)

---

## 1. Introduction & Football Context

Every spring, NFL draft boards are reshuffled around 40-yard dash times. Defensive ends running sub-4.60 generate buzz, while those in the 4.70s slide down boards. Yet on Sundays, straight-line speed rarely determines who consistently wins around the edge. 

Pass rushing is fundamentally curvilinear. To beat an offensive tackle setting vertically, an edge defender must win the outside shoulder and bend toward the quarterback. That turn requires generating substantial centripetal force at high speed. If a rusher cannot hold a tight arc, centrifugal force carries them wide into the tackle's recovery mirror, washing them past pocket depth. Coaches call this trait "bend"—the ability to drop the hips, flex at the ankle, and flatten to the passer without decelerating.

Evaluating bend before draft day has grown difficult as over 60% of top edge prospects now opt out of Combine agility testing (3-cone and short shuttle). However, nearly all run on-field positional drills. Chief among these is the Hoop Drill (`RUN_THE_HOOP_DRILL`), where rushers circle a ring of bags to demonstrate turning mechanics.

Using 10 Hz tracking from the 2023–2025 Combines and 44,002 NFL plays, this project introduces the **Edge Bend Index (EBI)**. By calculating path curvature and centripetal acceleration through the hoop, we quantify rusher flexibility, speed maintenance, and path discipline. 

We show that 40-yard dash times have no correlation with NFL sacks ($r = 0.101, p = 0.551$). In contrast, Combine hoop bend significantly predicts quick pressures ($<2.5$s) and sacks ($p = 0.018$), providing an objective on-field substitute for missing agility test data.

---

## 2. Kinematic Framework: Measuring Hoop Curvature

### 2.1 Differential Geometry of Rusher Paths
In the Combine Hoop Drill, prospects sprint around a circle of bags ($R_0 = 5.62$ yards) centered at $(x_0, y_0) = (48.557, 23.378)$.

We smooth 10 Hz coordinates $\mathbf{r}(t) = (x(t), y(t))$ using a 2nd-order Savitzky-Golay filter ($\Delta t = 0.1\text{s}$) to compute velocity $\mathbf{v}(t) = (\dot{x}(t), \dot{y}(t))$, scalar speed $v(t) = \sqrt{\dot{x}^2 + \dot{y}^2}$, and acceleration $\mathbf{a}(t) = (\ddot{x}(t), \ddot{y}(t))$.

Path curvature $\kappa(t)$ and turning radius $R(t)$ follow standard differential geometry:

$$\kappa(t) = \frac{|\dot{x}(t)\ddot{y}(t) - \dot{y}(t)\ddot{x}(t)|}{\left(\dot{x}(t)^2 + \dot{y}(t)^2\right)^{3/2}}, \quad R(t) = \frac{1}{\kappa(t)}$$

To hold an arc without drifting wide, a rusher must generate centripetal acceleration $a_c(t)$:

$$a_c(t) = v(t)^2 \kappa(t) = \frac{v(t)^2}{R(t)}$$

### 2.2 The Edge Bend Index (EBI)
We isolate three physical components from each hoop rep:
1. **Peak Centripetal Acceleration ($a_{c,\max}$):** Maximum lateral acceleration sustained at the apex of the arc.
2. **Speed Retention Ratio ($\text{SRR}$):** Fraction of entry speed maintained through the turn: $\text{SRR} = v(\kappa_{\max}) / v_{\text{entry}}$. Stiff rushers shed over half their speed to make the corner; fluid benders retain over 55%.
3. **Radial Drift Penalty ($\Delta R_{\text{drift}}$):** Average radial displacement outside the hoop boundary:

$$\Delta R_{\text{drift}} = \frac{1}{T} \int_0^T \max\left(0, \sqrt{(x - x_0)^2 + (y - y_0)^2} - R_0\right) dt$$

We combine these into the **Edge Bend Index (EBI)**, scaled from 0 to 100 across Combine edge participants:

$$\text{EBI} = \frac{a_{c,\max} \times \text{SRR}}{T_{\text{hoop}} \cdot \left(1 + 0.5 \Delta R_{\text{drift}}\right)}$$

---

### Figure 1: Hoop Drill Anatomy and Kinematics
![Figure 1: Hoop Drill Anatomy](https://raw.githubusercontent.com/abenih/nfl-big-data-bowl-2027/main/figures/figure_1_hoop_anatomy.png)

*Figure 1: (A) 2D tracking of Laiatu Latu (EBI: 82.9) versus a high-drift edge rusher around the 5.62-yard Combine hoop. (B) Velocity $v(t)$ and centripetal acceleration $a_c(t)$ profiles over time.*

---

## 3. Straight-Line Speed vs. Curvilinear Bend

Linking Combine tracking with regular-season play outcomes for qualified edge rushers (DE/OLB with $\ge 60$ pass rush snaps) reveals a direct contrast between straight-line testing and curvilinear movement (Figure 2):
- **40-Yard Dash vs. NFL Sack Rate:** $r = +0.101, p = 0.551$ ($R^2 = 0.010$). Straight-line sprint time has no predictive relationship with finishing sacks in the NFL.
- **Edge Bend Index vs. NFL Sack Rate:** $r = +0.336, p = 0.018$ ($R^2 = 0.113$). Rushers who sustain high centripetal acceleration while hugging the hoop finish sacks at a significantly higher rate on Sundays.

---

### Figure 2: Straight-Line Speed vs. Edge Bend
![Figure 2: The Combine Stopwatch Illusion](https://raw.githubusercontent.com/abenih/nfl-big-data-bowl-2027/main/figures/figure_2_combine_illusion.png)

*Figure 2: (A) 40-yard dash time versus NFL sack rate shows no relationship ($p = 0.55$). (B) The tracking-derived Edge Bend Index shows a significant positive correlation with sack rate ($p = 0.018$).*

---

## 4. Quick Pressures: Beating the 2.5-Second Threshold

NFL passes are released on average in 2.75 seconds, with quick concepts under 2.50 seconds. Late pressures often reflect secondary coverage or broken scrambles rather than clean pass-rush wins. For an edge rusher, the key efficiency metric is **Quick Pressure Rate** (pressures generated under 2.50 seconds from snap).

We estimated ordinary least squares (OLS) regression models controlling for player weight and 40-yard dash time:

$$\text{Quick Pressure Rate}_i = \beta_0 + \beta_1 (\text{40-Time}_i) + \beta_2 (\text{Weight}_i) + \beta_3 (\text{EBI}_i) + \epsilon_i$$

As shown in **Table 1**:
- In baseline testing (Model 1), 40-yard dash time is not statistically significant ($p = 0.541$).
- Adding EBI (Models 2 and 4) provides statistically significant explanatory power for both overall pressure rate ($\beta_3 = +0.168, p = 0.037$) and quick pressure rate ($\beta_3 = +0.086, p = 0.015$).
- For quick pressures, adding EBI doubles the model's adjusted $R^2$ from 0.103 to 0.201 over standard physical testing.

---

### Table 1: Multivariate Regression Models of NFL Pass Rush Efficiency

| Model | Spec | n | R² | Adj R² | EBI Coef (p) | 40-Time (p) |
|:---|:---|:---|:---|:---|:---|:---|
| M1: Baseline (PR) | 40-Time + Wt | 44 | 0.145 | 0.103 | N/A | -0.023 (.541) |
| M2: Augmented (PR) | 40 + Wt + EBI | 40 | 0.231 | 0.167 | +0.168 (.037*) | -0.014 (.700) |
| M3: Augmented (Sacks) | 40 + Wt + EBI | 40 | 0.111 | 0.037 | +0.024 (.207) | +0.012 (.165) |
| M4: Quick PR (<2.5s) | 40 + Wt + EBI | 40 | 0.262 | 0.201 | +0.086 (.015*) | -0.009 (.576) |

*Table 1: Regression models for qualified edge rushers ($n=40$). EBI provides statistically significant explanatory value for overall pressure rate ($p=0.037$) and quick pressures under 2.5s ($p=0.015$).*

---

### Figure 3: Time-to-Pressure and Quick Pressure Rates
![Figure 3: Time-to-Pressure Analysis](https://raw.githubusercontent.com/abenih/nfl-big-data-bowl-2027/main/figures/figure_4_time_to_pressure.png)

*Figure 3: (A) Quick pressure rates across EBI quartiles; top-quartile rushers generate quick pressures at nearly twice the rate of bottom-quartile rushers ($p=0.015$). (B) Cumulative distribution of time to pressure.*

---

## 5. Film & Tracking Translation: Laiatu Latu Case Study

To examine how drill mechanics translate to game action, we analyzed 10 Hz game tracking from 2024 first-round pick **Laiatu Latu** (EBI: 82.9):
- **Combine Hoop Rep:** Latu reached an apex curvature $\kappa = 0.21\text{ yd}^{-1}$ (turning radius $R = 4.76$ yards) while maintaining $v = 6.40\text{ m/s}$ (14.3 mph).
- **In-Game Strip Sack (Week 3 vs. Chicago Bears, Play 3940):** Rushing against the left tackle, Latu flattened a curvilinear arc with curvature $\kappa = 0.20\text{ yd}^{-1}$, sustained speed through contact, and stripped Caleb Williams at 2.40 seconds from snap (Figure 4).

The turning radius, curvature, and velocity retention observed in Indianapolis closely matched his in-game sack track on Sunday.

---

### Figure 4: Combine Drill vs. In-Game Strip Sack
![Figure 4: Sunday Translation](https://raw.githubusercontent.com/abenih/nfl-big-data-bowl-2027/main/figures/figure_5_sunday_translation.png)

*Figure 4: Comparison of Laiatu Latu's Combine Hoop Drill path (left) with his rush trajectory on a strip sack of Caleb Williams (right). Turning curvature and speed retention match between drill and game play.*

---

## 6. Positional Nuance: Edge Rushers vs. Interior Linemen

A sound football metric must respect schematic roles. Interior linemen (DT/NT) operate in congested gaps where wins depend on mass, leverage, and shedding double teams. Perimeter edge rushers (DE/OLB) win primarily in space along the tackle's outside shoulder.

When segmenting the cohort by alignment, this tactical distinction is evident (Figure 5):
- For **Perimeter Edge Rushers**, EBI correlates strongly with pressure rate ($r = +0.357, p = 0.012$).
- For **Interior Linemen (iDL)**, EBI shows zero correlation with pressure rate ($r = -0.043, p = 0.835$).

The lack of correlation among interior rushers confirms that EBI specifically reflects the biomechanics of turning the corner against an offensive tackle.

---

### Figure 5: Scheme and Alignment Specificity
![Figure 5: Scheme Specificity](https://raw.githubusercontent.com/abenih/nfl-big-data-bowl-2027/main/figures/figure_6_scheme_specificity.png)

*Figure 5: EBI predicts pass rush pressure rate for perimeter edge rushers (left, $p=0.012$), but shows no relationship for interior defensive linemen (right, $p=0.835$).*

---

## 7. Scouting Application: The Speed-Bend Matrix

Plotting 40-yard sprint times against the Edge Bend Index creates an actionable 2x2 framework for draft boards (Figure 6):

- **Quadrant I: High Speed, High Bend (Top Tier):** Prospects like Laiatu Latu, Chop Robinson, and Jared Verse convert athletic burst into elite cornering, projecting as immediate impact rushers.
- **Quadrant II: Modest 40, High Bend (Draft Steals):** Day 2 and Day 3 picks like Austin Booker (Round 5) and Braden Fiske (Round 2) slipped due to modest 40 times (4.75s+). Yet top-tier hoop bend allowed them to outproduce their draft slots in pressures per snap.
- **Quadrant III: Fast 40, Low Bend (Bust Risk):** Straight-line athletes with high radial drift who struggle to corner against NFL tackles, frequently washing past the pocket.
- **Quadrant IV: Modest 40, Low Bend (Base Ends):** Early-down run stoppers lacking perimeter burst and cornering flexibility.

---

### Figure 6: The 2x2 Edge Rusher Evaluation Matrix
![Figure 6: The 2x2 Prospect Matrix](https://raw.githubusercontent.com/abenih/nfl-big-data-bowl-2027/main/figures/figure_3_prospect_matrix.png)

*Figure 6: Prospect evaluation framework segmenting edge rushers by 40-yard dash speed and Edge Bend Index, with bubble size proportional to NFL pass rush volume.*

---

### Table 2: Combine Edge Rush Scouting Leaderboard (Selected Prospects)

| Player | Draft | 40-Time | Hoop Time | Apex v | SRR | EBI | NFL Sacks | Tier |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Chop Robinson | 2024, #21 | 4.48s | 7.20s | 2.27 m/s | 55.4% | 100.0 | 11.0 | Tier 1: Elite Bend |
| Laiatu Latu | 2024, #15 | 4.64s | 7.40s | 2.27 m/s | 52.8% | 82.9 | 13.5 | Tier 1: Elite Bend |
| Jared Verse | 2024, #19 | 4.58s | 7.50s | 2.23 m/s | 59.0% | 80.5 | 15.0 | Tier 1: Elite Bend |
| Austin Booker | 2024, #144 | 4.79s | 6.40s | 2.29 m/s | 56.8% | 82.4 | 12.5 | Tier 2: Value Steal |
| Braden Fiske | 2024, #39 | 4.78s | 7.20s | 2.25 m/s | 52.8% | 86.0 | 15.0 | Tier 1: Elite Bend |

*Table 2: Combine prospects ranked by Edge Bend Index alongside physical testing and regular-season NFL production.*

---

## 8. Coaching and Training Applications

Hoop drill tracking provides actionable diagnostics for coaching and performance staffs:
- **Mobility Screening:** Excessive radial drift signals limited ankle dorsiflexion or hip rotation, guiding targeted rookie mobility interventions.
- **Curvilinear Resistance:** Training with lateral tether resistance around a 5-yard arc builds the adductor and ankle strength needed to resist centrifugal force.
- **Fatigue Monitoring:** Tracking weekly changes in hoop curvature detects lower-body fatigue before soft-tissue injuries occur.

---

## 9. Conclusion

With prospects frequently opting out of standardized agility drills, positional sensor tracking offers an objective alternative to evaluate movement. The Edge Bend Index isolates the biomechanics of bend from the Combine Hoop Drill, replaces missing 3-cone data, and predicts quick pressures and sacks. In a passing league decided in under 2.5 seconds, bend separates winning rushers from straight-line sprinters.

---

## Appendix & Reproducibility
All data processing, kinematic calculations, statistical regressions, and figures are fully reproducible within the companion Kaggle notebook `abenih/notebook09af53edae`. Raw tracking data was provided by the NFL Big Data Bowl 2027 under CC BY-NC 4.0 license.
