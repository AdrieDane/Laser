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
| Windows 11 Pro | 10.0.26200, **Smart App Control: aan** | zie §10 |
| Python | 3.14.6 (`py -3.14`) | was al aanwezig |
| Git | 2.54 | aanwezig |
| R (alleen voor optionele `tools/*.R`) | 4.6.0 in `C:\Program Files\R\R-4.6.0` (niet in PATH) | aanwezig; xcms/MSnbase/jsonlite geïnstalleerd maar xcms laadt niet (§10), CAMERA/IDSL.UFA nee |
| WSL2 Ubuntu 24.04 | Python 3.12.3, venv in `~/laser-venv` | gebruikt voor de volledige suite (§7) |

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
De suite duurt ongeveer 3 minuten. Voor de volledige suite inclusief IsoSpecPy, zonder
consolevensters, gebruik WSL (§7).

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

### Testsuite

Na de Windows-fixes uit §8 (9 november 2026):

| Omgeving | Geslaagd | Gefaald | Overgeslagen |
|---|---|---|---|
| Windows native (`.venv`, Python 3.14) | 1070 | 33 | 12 |
| WSL2 Ubuntu (`~/laser-venv`, Python 3.12) | 1102 | 1 | 12 |

- **Windows, 33 fouten:** allemaal IsoSpecPy, waarvan Smart App Control de DLL
  blokkeert (`test_complete_isotopes`, `test_lipid_formulas`, 8 in `test_joint_inference`).
  Dit betreft alleen de optionele complete-isotopenmodus. In WSL slagen ze allemaal.
- **WSL, 1 fout:** `test_empirical_candidates::test_provably_empty_mass_intervals…`
  eist bit-identieke floats tussen twee varianten van dezelfde berekening. Onder
  Linux verschilt `ref_rt` 1e-16 (afrondingspad van numpy); op Windows met numpy
  2.4.4 slaagt de test. Dit is geen wetenschappelijke fout. De test is bewust niet
  versoepeld, omdat upstream deze bit-exactheid bewaakt.
- Vóór de fixes: Windows 1062/42/11 (incl. de 33 IsoSpecPy-fouten) en WSL 1099/4/12.

**Ruff:** 3 stijlmeldingen (E702, puntkomma's) in `app/tests/test_untargeted_grid.py`
(upstream-code, niet functioneel).

### Oplossingen tijdens de installatie (alleen pakketversies)

| Probleem | Oplossing |
|---|---|
| pandas 3.1.1: `timestamps.pyd` geblokkeerd, alle 122 testbestanden falen | pandas **3.0.2** |
| scikit-learn 1.9.1: `_vector_sentinel` geblokkeerd | scikit-learn **1.8.0** |
| numpy 2.5.4: 1-ULP-verschil in `ref_rt` | numpy **2.4.4** (de versie die upstream gebruikt) |
| `.exe`-launchers geblokkeerd | `python -m standard_cli` / `python -m streamlit` |

## 7. WSL2-omgeving (volledige suite incl. IsoSpecPy)

In WSL geldt Smart App Control niet. Er is geen sudo nodig: de venv wordt
gebootstrapt met de pip-wheel die bij de Windows-Python wordt meegeleverd.

```bash
python3 -m venv --without-pip ~/laser-venv
W=/mnt/c/Users/Adrie/AppData/Local/Python/pythoncore-3.14-64/Lib/ensurepip/_bundled/pip-26.1.2-py3-none-any.whl
~/laser-venv/bin/python $W/pip install --no-index $W
cd /mnt/c/Users/Adrie/Laser
~/laser-venv/bin/python -m pip install -e ".[dev,calibration]" -c docs/constraints-windows-lipidlive.txt
~/laser-venv/bin/python -m pytest -q -p no:cacheprovider
```
Vanuit Windows: `wsl.exe -d Ubuntu -e bash -lc '…'`. De WSL-suite opent geen
Windows-consolevensters. Hij is wel iets trager (~3,5 min), door bestandstoegang via `/mnt/c`.

## 8. Windows-fixes in de code (geen wetenschappelijke logica)

| Wijziging | Effect |
|---|---|
| `application/annotation_jobs.py`: worker krijgt `PYTHONIOENCODING=utf-8` | **App-bug:** de untargeted-analyse crashte op Windows zodra een voortgangsbericht met `↔` naar het log ging (cp1252) |
| idem: `CREATE_NO_WINDOW` in plaats van `DETACHED_PROCESS` | Worker blijft los van de serverterminal. Console-programma's die de worker start, openen geen zichtbare vensters meer |
| idem: `status.json` lezen met korte retry bij `PermissionError` (alleen Windows) | Voorkomt sporadische fouten als de GUI leest terwijl de worker de status atomisch vervangt |
| `output_io.py`: `os.replace` met korte retry bij `PermissionError` (alleen Windows) | Idem aan de schrijfkant; atomiciteit ongewijzigd |
| JSON-bestanden van de app expliciet als UTF-8 lezen (9 modules) | **App-bug:** de methode-instellingen (`×`, `µ`, `Å`) werden op Windows bij elk opslaan/laden verminkt (`Ã—`, `Âµ`) |
| `presentation/untargeted_mode/__init__.py`: JS/CSS als UTF-8 lezen | De untargeted viewer kon op Windows niet laden |
| `tools/freeze_untargeted_baseline.py`: manifest-sleutels met `/` | Fingerprints identiek op alle platforms |
| Tests: `.run()` na het invullen van de projectmap (3×) | Nieuwere Streamlit weigert terecht klikken op een nog uitgeschakelde knop (ook onder Linux) |
| Tests: POSIX-paden, UTF-8 en symlinktest overslaan zonder symlinkrechten | Test-portabiliteit |

Controle: de LASER-CLI-uitvoer op de synthetische data is byte-identiek aan die van vóór de fixes.

## 9. Echte data: `26D0111_BarthTest`

In deze map (Nextcloud) zitten geen XCMS-peakgrouplijsten, dus LASER kan haar niet direct gebruiken:

- `Data/Processed/mexp_RP_{Pos,Neg}.Rdata` is een `MsExperiment` (verwijzingen
  naar ruwe spectra op `G:\…`, niet aanwezig). `xexp_is_*.Rdata` bevat alleen de
  detectie van interne standaarden. De volledige featuretabel
  (`xexpfinal_<mode>.Rdata` in `lipidsencore`) bestaat nog niet.
- Alleen RP; 5 Control, 5 Barth, 6 QC en 6 blanco's per polariteit.
- Native R kan xcms niet laden: Smart App Control blokkeert
  `fs.dll` in `AppData\Local\R\win-library\4.6`.

LASER heeft per modus nodig: `peakgrouplist_RP_Pos.txt` / `_Neg.txt` (tab-gescheiden;
`name`, `mzmed`, `rtmed` in minuten, eventueel `mzmin/mzmax/rtmin/rtmax/npeaks`,
`Group_*`-tellingen, dan de sample-intensiteiten). Die kunnen uit een gegroepeerd
`XcmsExperiment` worden gemaakt (`featureDefinitions` + `featureValues`, RT ÷ 60).

## 10. Beperkingen en openstaande punten

1. **Smart App Control** blokkeert ongesigneerde DLL's en exe's zonder reputatie,
   ook in R-bibliotheken. Na upgrades kunnen nieuwe blokkades opduiken.
   Uitzetten is systeembreed en op Windows 11 niet zonder herinstallatie terug te
   draaien, dus dat is niet gedaan.
2. **IsoSpecPy** werkt alleen in WSL (§7).
3. **R-tools** (`tools/*.R`): R 4.6.0 staat niet in PATH. CAMERA en IDSL.UFA ontbreken,
   en xcms laadt native niet (zie 1). Alleen relevant voor upstream-onderzoeksbenchmarks.
4. **Geen echte testdata in LASER-formaat.** De validatie is alleen synthetisch (§9).
5. `AGENTS.md` (225 kB) bevat instructies voor de upstream-onderzoeksagent
   (optimalisatiecycli op privédatasets van het CFM). Die zijn grotendeels niet van
   toepassing op deze fork, maar de beperkingen (geen ruwe data in Git, standaardinstellingen
   niet wijzigen) zijn verstandig om aan te houden.
