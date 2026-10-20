# Mechanical Design Explorer

Interactive engineering tools for the questions mechanical, quality and manufacturing engineers ask every day: **will this beam or shaft survive, will these parts assemble and which tolerance should be tightened, and where does a pump and pipe system lose pressure or cavitate.**

**[Open the live app](https://mechanical-design-explorer-dt92wtfsjydzt3htt2zmkg.streamlit.app/)** · Built with Python, NumPy, Pandas, Plotly and Streamlit · Free hosting, so the first load may take a few seconds.

> **For learning and screening, not certified design.** Example inputs are illustrative, not from real projects. Pump data, loss coefficients, material properties and tolerance values are generic textbook-style numbers.

## What it does

Three chapters in one app, each switchable between SI and US units. Every input is validated, and impossible values (zero diameter, negative span, a minimum size above the maximum, and so on) give a clear message instead of a crash.

### 1. Beam & shaft stress
*Question: does this part carry the load with enough margin?*

- Sliders for load size and position, span, support (simply supported or cantilever), cross-section (rectangle, solid round, hollow round) and size, material (dropdown with E, yield and ultimate strength), and torque for shafts.
- Shows the part coloured blue (low stress) to red (near yield) on its deformed shape, shear and moment diagrams, deflection, safety factor and a pass or fail against the required safety factor.
- Optional fatigue check using the Goodman relation.
- Choosing a stronger material changes the safety factor and the colour scale, not the stress pattern.

![Stress map on the deformed shape](docs/beam-shaft-stress.png)
![Shear and moment diagrams](docs/beam-shaft-diagrams.png)

### 2. Pump & pipe system
*Question: where does this system lose pressure, what flow will the pump deliver, and will it cavitate?*

- Sliders for suction and discharge pipe diameter and length, tank levels, bends, valves, fittings, pump speed and water temperature. A pipe material dropdown sets roughness.
- Draws the pipe layout coloured by pressure or velocity, finds the operating point (pump curve against system curve), highlights the section with the largest pressure drop as the bottleneck, and flags cavitation risk by comparing NPSH available with NPSH required.
- Method: Darcy-Weisbach friction with a laminar or Colebrook-White friction factor, fitting losses from loss coefficients, a quadratic pump curve scaled with the affinity laws. This is system-curve analysis, **not CFD**.

![Pipe layout coloured by pressure](docs/pump-pipe-layout.png)
![Pump curve against system curve](docs/pump-pipe-curves.png)

### 3. Tolerance stack-up & GD&T
*Question: will these parts assemble, and which dimension should be tightened?*

- Dimensions and tolerances in an editable table (nominal, +tol, −tol, direction). Compares **worst-case, RSS and Monte Carlo**, shows the gap histogram against the spec limits with Cpk and the percentage out of spec, and ranks which dimension contributes most to the variation.
- A second tab shows a **GD&T position tolerance zone** for a hole or pin, with bonus tolerance at MMC drawn to scale, a pass or fail, the virtual condition, and the same part judged at RFS for comparison.
- A learning tool based on ASME Y14.5 concepts. **It is not a certified interpretation of the standard.**

![Gap histogram with spec limits and Cpk](docs/tolerance-histogram.png)
![Position tolerance zone with bonus tolerance](docs/gdt-position-zone.png)

*The images are renders of the app's own charts with illustrative inputs.*

## How the results are checked

206 automated tests (`python -m pytest`) cover the calculations, the input validation and the app itself. Calculations are pure functions with no Streamlit dependency, so they are tested directly against textbook formulas and hand calculations.

| Area | Tests | What is checked |
|---|---|---|
| Beam & shaft | 48 + 5 app | Reactions, shear and moment for centre and off-centre loads (PL/4); deflection against PL³/48EI and PL³/3EI; the small 1D finite-element model against the closed form; section properties; torsion τ = Tc/J; von Mises; endurance limit and Goodman; combined bending and torsion hand calculation; material changes the safety factor but not the stress pattern; invalid loads and sections; figures build |
| Tolerance stack-up & GD&T | 53 + 5 app | Worst-case, RSS (√Σt²) and Cpk against hand values for a worked stack; asymmetric tolerances; contribution shares; Monte Carlo against the analytic mean, σ and normal-tail fraction (statistical checks with fixed seed); hole and pin bonus tolerance, zone and virtual condition; RFS; boundary and out-of-size cases; invalid rows, spec limits and sample counts |
| Pump & pipe | 49 + 6 app | Laminar f = 64/Re; Colebrook against the Moody chart and Swamee-Jain; Darcy-Weisbach and minor-loss hand values; an independent hand calculation of the system head; affinity laws; analytic operating point; NPSH available and cavitation thresholds; water vapour pressure; energy balance along the layout; trends with speed, temperature and pipe size; weak pump (no operating point); transitional flow flag; invalid layouts, pumps and speeds |
| Shared code | 40 | Unit conversions and round trips (20), validation helpers (11), chapter auto-discovery (5), app smoke tests (4) |

The app tests use Streamlit's AppTest to load each chapter, change inputs, switch units and confirm that bad inputs show a message. **Not covered:** testing in a real browser, and cell-by-cell editing of the tolerance table (the table-to-calculation conversion is unit-tested).

## Limitations

- **Beam & shaft:** linear-elastic Euler-Bernoulli beam with one point load; self-weight ignored. Transverse shear, stress concentrations, buckling and contact stresses are not included. The Goodman check uses a simplified endurance limit (k · 0.5 · Su, capped at 700 MPa) and applies the same load ratio to bending and torque. The 1D finite-element model is a cross-check, not a general FEA tool.
- **Pump & pipe:** one series path with one pipe material; water only at sea-level pressure. Loss coefficients are rounded textbook values and real fittings vary. The pump curve is a quadratic through the shutoff and design points; the NPSH-required shape is an assumption. Pipe elevation changes in the colour map, pump efficiency and power, water hammer, two-phase flow and parallel branches are not modelled. Flow between Re 2300 and 4000 is flagged as transitional and its friction factor is only a guide.
- **Tolerance stack-up:** one-dimensional linear stacks. Each tolerance band is treated as ±kσ of a normal distribution (k selectable, default 3) and Monte Carlo samples are not truncated at the limits. Process drift, correlated dimensions, non-normal distributions and 2D or 3D stacks are not modelled.
- **GD&T:** a single feature of size with one position zone at MMC or RFS. Datum reference frames and datum shift, composite or projected tolerance zones, orientation and form controls, and LMC are not modelled.

## Run it locally

```bash
pip install -r requirements.txt     # streamlit>=1.50, plotly, pandas, numpy, pytest
streamlit run app.py                # all chapters
streamlit run chapters/pump_pipe.py # one chapter on its own
python -m pytest                    # the test suite
```

## Add a chapter

`app.py` finds every file in `chapters/` automatically, so a new chapter needs **no edits to `app.py`**.

1. Copy `chapters/_template.py` to `chapters/<your_chapter>.py` (no leading underscore).
2. Set `TITLE` and `DESCRIPTION`, put the calculations in pure functions (SI in, SI out, no Streamlit) at the top of the file, and write the interface in `render()`.
3. Import only from `common/` and never from another chapter. Raise `ValidationError` (from `common/validation.py`) for impossible inputs.
4. Run it alone with `streamlit run chapters/<your_chapter>.py`, add tests that import it with `from chapters import <your_chapter>`, then reload the app: it appears in the chapter dropdown.

## Project layout

```
app.py              base shell: chapter dropdown, units, disclaimer
chapters/           one file per chapter (beam_shaft, tolerance_stack, pump_pipe), plus _template.py and a small example
common/             units (SI and US), validation, plot theme, shared UI helpers, chapter discovery
tests/              pytest and Streamlit AppTest suites
docs/               chart images used in this README
```

## About

Built by [Gaurav Haresh Patil](https://gp26902.github.io), industrial and mechanical engineer (M.S. Industrial Engineering, B.E. Mechanical Engineering). More projects are on the [portfolio](https://gp26902.github.io).
