# LASER lokaal (branch `lipidlive`): installatie en testresultaten

Getest op 9 november 2026 op commit `58fad61` (gelijk aan `upstream/main`).

## 1. Wat LASER is

LASER Standard Lipidomics annoteert lipiden in conventionele XCMS-featuretabellen
(LC-MS, RP- en NP-chromatografie, positieve en negatieve ionisatie). Het is een
Python-applicatie (Streamlit-GUI plus een kleine CLI). MATLAB is niet nodig: de
MATLAB-pipeline is in Python nagebouwd (`app/matlab_annotation/vendor`).

| Onderdeel | Map | Functie |
|---|---|---|
| GUI | `app/app.py`, `app/presentation/` | Streamlit, vijf modules: Calibration, Annotation, Untargeted analysis, en twee viewers voor opgeslagen resultaten |
| CLI | `app/standard_cli.py` | `laser-standard`: headless annotatie |
| Workflow/jobs | `app/application/` | Achtergrondjobs (losse Python-processen), logs onder `.laser/jobs/` |
| Engine 1 | `app/matlab_annotation/` | "Automatic MATLAB engine" (standaard): RT-uitlijning via ¹³C-paren en RTDB, gefaseerde annotatie, <3 ppm |
| Engine 2 | `app/annotator/` | "LASER reference matcher": matching tegen `reference_data/4d_lipid_reference.csv` met MATLAB-policy |
| Untargeted | `app/standard_lipidomics/` | Statistiek, adduct/isotoop/formule-evidence (veel experimentele opties, standaard uit) |
| Referentiedata | `reference_data/` | Lipidcatalogus, RT-databases, policies (meegeleverd, ca. 5 MB) |
| Onderzoekstools | `tools/` | Evaluatie-/optimalisatiescripts van de upstream-ontwikkelaars, deels in R |

**Invoer:** een projectmap met tab-gescheiden XCMS-peakgrouplijsten
`peakgrouplist_{RP,NP}_{Pos,Neg}.txt` (in de map zelf of in `XCMS/`), met kolommen
`name`, `mzmed`, `rtmed` (RT in **minuten**), optioneel `mzmin/mzmax/rtmin/rtmax/npeaks`,
`Group_*`-tellingen en daarna de sample-intensiteiten. Optioneel `metadata/`
met sample-groepwerkboeken en `internal_standards.csv`.

**Uitvoer (CLI):** CSV-tabellen (`automatic_annotations`, `identified_lipids`,
`all_candidates`, `policy_audit`, …) en `annotated_input_files/*_annotated.txt`:
de originele lijsten met alleen de kolommen 2/3 (`laser_annotation`, `adduct`) toegevoegd.
De GUI schrijft naar `Calibration/`, `LASER/`, `Untargeted/` en `Archive/` in de projectmap.

**Voorbeelddata:** niet meegeleverd. De tests gebruiken synthetische data.
Voor een end-to-end-test staat in `tools/lipidlive/` een generator voor een
volledig synthetisch project (zie §4).

**Externe systemen:** de kern-app heeft geen netwerk of externe software nodig.
Optioneel en experimenteel: msbuddy (aparte Python-omgeving), IsoSpecPy (complete
isotopenmodus), R-scripts in `tools/` (CAMERA, IDSL.UFA, xcms/mzR), SIRIUS en MZmine.
Geen daarvan is nodig voor de standaard workflow.

## 2. Benodigde software

| Software | Versie getest | Status op deze pc |
|---|---|---|
| Windows 11 Pro | 10.0.26200, **Smart App Control: aan** | zie §6 |
| Python | 3.14.6 (`py -3.14`) | was al aanwezig |
| Git | 2.54 | aanwezig |
| R (alleen voor optionele `tools/*.R`) | 4.6.0 in `C:\Program Files\R\R-4.6.0` (niet in PATH) | aanwezig; xcms/MSnbase/jsonlite ja, CAMERA/IDSL.UFA nee |
| WSL2 Ubuntu 24.04 | Python 3.12.3 zonder `python3-venv` | aanwezig, niet gebruikt |

Alle Python-pakketten staan in een geïsoleerde `.venv` in de repo (staat in
`.gitignore`). Exacte versies: [`constraints-windows-lipidlive.txt`](constraints-windows-lipidlive.txt).

## 3. Installatie

Voer uit in PowerShell vanuit `C:\Users\Adrie\Laser`:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev,calibration]" -c docs\constraints-windows-lipidlive.txt
```

Gebruik op deze pc **niet** `Setup (Windows).bat`: dat installeert zonder
constraints de nieuwste pandas (3.1.1), en daarvan blokkeert Smart App Control
een DLL (zie §6). Als die `.venv` al bestaat met de juiste versies, werkt
`Launch LASER Standard.bat` wel, omdat `pip install -e .` dan niets upgradet.

## 4. Starten

Gebruik altijd `python -m …`. De door pip gegenereerde `.exe`-launchers
(`laser-standard.exe`, `streamlit.exe`) worden door Smart App Control geblokkeerd.

**GUI (alleen lokaal):**
```powershell
.venv\Scripts\python.exe -m streamlit run app\app.py --server.address 127.0.0.1 --server.port 8085 --browser.gatherUsageStats false
```
Open daarna http://localhost:8085. (De `.bat`-launcher luistert op `0.0.0.0`, dus op
alle netwerkinterfaces; voor lokaal experimenteren is `127.0.0.1` veiliger.)

**CLI:**
```powershell
.venv\Scripts\python.exe -m standard_cli <projectmap> --engine matlab --phase both --output <uitvoermap>
.venv\Scripts\python.exe -m standard_cli <projectmap> --engine laser --output <uitvoermap>
```

**Synthetische end-to-end-rooktest** (geen echte data; m/z en RT uit de meegeleverde
referentie met 1 ppm/0,03 min ruis, plus ¹³C-isotooppieken, 300 ruisfeatures en
willekeurige intensiteiten):
```powershell
$T = "$env:TEMP\laser_smoke"
.venv\Scripts\python.exe -I tools\lipidlive\make_synthetic_project.py . $T\synthetic_project
.venv\Scripts\python.exe -m standard_cli $T\synthetic_project --engine matlab --phase RP --output $T\out_matlab
.venv\Scripts\python.exe -m standard_cli $T\synthetic_project --engine laser --output $T\out_laser
.venv\Scripts\python.exe -I tools\lipidlive\check_synthetic_output.py $T out_matlab
.venv\Scripts\python.exe -I tools\lipidlive\check_synthetic_output.py $T out_laser
```

## 5. Tests

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -m ruff check app
```
De suite duurt ongeveer 3 minuten en start veel losse workerprocessen. Daardoor
kunnen er kort consolevensters openen en sluiten.

## 6. Resultaten (9 november 2026)

### Synthetische end-to-end-run (alleen RP, 307 Pos- en 289 Neg-lipiden geplant)

| Engine | Tijd | Annotaties Pos / Neg | Op ruisfeatures | Zelfde lipide+adduct op het geplante feature |
|---|---|---|---|---|
| MATLAB (standaard) | 234 s | 53 / 47 | 0 / 0 | 52/53, 43/47 |
| LASER matcher | 8 s | 130 / 124 | 0 / 3 (¹³C-isotooppieken) | 125/130, 118/124 |

In alle uitvoerbestanden zijn de originele kolommen byte-identiek. Dit is een
**rooktest** (draait het, en is de uitvoer plausibel), **geen nauwkeurigheidsmeting**.
De synthetische data is te idealistisch en de MATLAB-engine accepteert bewust alleen
lipiden met voldoende RT-ondersteuning. Zonder ¹³C-isotooppieken faalt de
MATLAB-engine met `zero-size array to reduction operation maximum`, omdat er geen
RT-uitlijningsankers zijn. Echte XCMS-data bevat die isotooppieken wel; een
duidelijkere foutmelding zou nuttig zijn.

**GUI:** Streamlit start, `/_stcore/health` geeft `ok` en de startpagina geeft HTTP 200.

### Testsuite: 1062 geslaagd, 42 gefaald, 11 overgeslagen (+91 subtests geslaagd)

| Oorzaak | Aantal | Toelichting |
|---|---|---|
| IsoSpecPy-DLL geblokkeerd door Smart App Control | 33 | `test_complete_isotopes`, `test_lipid_formulas`, 7 van `test_joint_inference`. Alleen voor de optionele complete-isotopenmodus; de code eist exact IsoSpecPy 2.6.1 |
| Streamlit AppTest: "Cannot update a disabled button" | 3 | `test_standard_project_modules`, 2× `test_untargeted_evidence`; vermoedelijk een verschil met Streamlit 1.67 |
| Padscheidingsteken `\` in plaats van `/` | 2 | `test_untargeted_evaluation_contract`: manifest-sleutels zijn op Windows niet POSIX |
| Symlink vereist adminrechten of Developer Mode (WinError 1314) | 1 | `test_joint_inference::test_runtime_freeze…` |
| Bestandsvergrendeling (`PermissionError` op `status.json`) | 1 | `test_full_feature_groups`: Windows staat geen gelijktijdige vervanging toe |
| Bestand gelezen als cp1252 in plaats van UTF-8 | 1 | `test_untargeted_mode_view` |
| Streamlit-numeriek veld geeft `14.0` in plaats van `14` | 1 | `test_acquisition_method` |

De standaard-annotatietests (`test_matlab_annotation`, `test_standard_lipidomics`,
`test_matching`, `test_standard_cli` en dergelijke) slagen allemaal.

**Ruff:** 3 stijlmeldingen (E702, puntkomma's) in `app/tests/test_untargeted_grid.py`.
Dit is upstream-code en niet functioneel.

### Oplossingen die tijdens de installatie zijn toegepast (geen codewijzigingen)

| Probleem | Oplossing |
|---|---|
| pandas 3.1.1: `timestamps.pyd` geblokkeerd, alle 122 testbestanden falen | pandas **3.0.2** |
| scikit-learn 1.9.1: `_vector_sentinel` geblokkeerd | scikit-learn **1.8.0** |
| numpy 2.5.4: 1-ULP-verschil in `ref_rt` (`test_empirical_candidates`) | numpy **2.4.4** (de versie die upstream gebruikt) |
| `.exe`-launchers geblokkeerd | `python -m standard_cli` / `python -m streamlit` |

## 7. Beperkingen en openstaande punten

1. **Smart App Control** blokkeert ongesigneerde DLL's en exe's zonder
   reputatie. Er kunnen ook later geblokkeerde bestanden opduiken, bijvoorbeeld na
   een `pip upgrade`. Uitzetten is een systeembrede beslissing; op Windows 11 kan
   het daarna niet zonder herinstallatie weer worden aangezet. Dat is dus niet gedaan.
2. **IsoSpecPy / complete-isotopenmodus** werkt niet native. Alternatief: WSL2
   Ubuntu (daar geldt Smart App Control niet). Daarvoor is `sudo apt install python3-venv`
   in WSL nodig, of een gebruikersinstallatie van `uv`.
3. **Windows-portabiliteit:** 9 tests falen door pad-, codering-, symlink- en
   lock-verschillen. Dit raakt de standaard-annotatieworkflow niet. Upstream-CI
   draait wel op Windows; of deze tests daar ook falen is niet nagegaan.
4. **R-tools** (`tools/*.R`): R 4.6.0 staat niet in PATH. CAMERA en IDSL.UFA ontbreken.
   Alleen relevant voor het reproduceren van de upstream-onderzoeksbenchmarks.
5. **Geen echte voorbeelddata.** De validatie is alleen synthetisch.
6. `AGENTS.md` (225 kB) bevat instructies voor de upstream-onderzoeksagent
   (optimalisatiecycli op privédatasets van het CFM). Die zijn grotendeels niet van
   toepassing op deze fork, maar de beperkingen (geen ruwe data in Git, standaardinstellingen
   niet wijzigen) zijn verstandig om aan te houden.

Nieuwe, nog niet gecommitte bestanden: dit document,
`docs/constraints-windows-lipidlive.txt` en `tools/lipidlive/`.
