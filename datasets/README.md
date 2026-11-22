# Datasets

This folder holds the data used by the IoV-BERT-IDS pipeline (`run_pipeline.py`). It contains four datasets in two groups:

| Group | Folder | Data type | Format |
|---|---|---|---|
| In-vehicle (CAN bus) | `car_hacking/` | CAN frames from one vehicle | CSV / TXT |
| In-vehicle (CAN bus) | `ivn/` | CAN frames from three vehicles | CSV |
| Extra-vehicle (IP network) | `botiot/` | IoT network packets | PCAP |
| Extra-vehicle (IP network) | `cicids/` | Enterprise network packets + labelled flows | PCAP + CSV |

```
datasets/
├── botiot/
│   ├── DDoS/ddos_1.pcap
│   ├── DoS/dos_1.pcap
│   ├── Normal/normal_1.pcap
│   ├── Reconnaissance/reconnaissance_1.pcap
│   └── Theft/theft_1.pcap
├── car_hacking/
│   ├── DoS_dataset.csv
│   ├── Fuzzy_dataset.csv
│   ├── gear_dataset.csv
│   ├── RPM_dataset.csv
│   └── normal_run_data.txt
├── cicids/
│   ├── Monday-WorkingHours.pcap
│   ├── AttackDay-WorkingHours.pcap
│   └── AttackDay-WorkingHours.pcap_ISCX.csv
└── ivn/
    ├── Sonata/{Flooding,Fuzzy,Malfunction,Normal}_sonata.csv
    ├── Soul/{Flooding,Fuzzy,Malfunction,Normal}_soul.csv
    └── Spark/{Flooding,Fuzzy,Malfunction,Normal}_spark.csv
```

---

## 1. `car_hacking/`: Car-Hacking dataset

**Source:** Hacking and Countermeasure Research Lab (HCRL), Korea University.
<https://ocslab.hksecurity.net/Datasets/car-hacking-dataset>

> H. M. Song, J. Woo, H. K. Kim, "In-vehicle network intrusion detection using deep convolutional neural network," *Vehicular Communications*, vol. 21, 2020.

CAN bus traffic recorded through the OBD-II port of a real vehicle, with attack messages injected.

**Format (CSV):** `Timestamp, CAN ID, DLC, DATA[0] … DATA[7], Flag`

```
1478246049.427327,0260,8,23,19,48,11,6c,cf,a5,dd,R
```

| Field | Meaning |
|---|---|
| Timestamp | Unix time in seconds |
| CAN ID | Arbitration ID in hex (e.g. `0260`) |
| DLC | Data length code: number of data bytes (0–8) |
| DATA[0–7] | Payload bytes in hex |
| Flag | `R` = normal message, `T` = injected attack message |

`normal_run_data.txt` holds attack-free traffic in a text format:

```
Timestamp: 1479121434.000187        ID: 018f    000    DLC: 8    8b e1 cc cd 48 7f 18 5d
```

| File | Attack | Normal (R) | Attack (T) |
|---|---|---|---|
| `DoS_dataset.csv` | DoS: high-priority ID `0000` flooded | 50,808 | 14,399 |
| `Fuzzy_dataset.csv` | Fuzzy: random IDs and payloads | 57,867 | 7,408 |
| `gear_dataset.csv` | Spoofing of the gear-position ID | 61,011 | 4,295 |
| `RPM_dataset.csv` | Spoofing of the engine-RPM ID | 60,797 | 4,507 |
| `normal_run_data.txt` | None | ~20,300 | 0 |

**Used in:** Experiment 2 (in-vehicle intrusion detection).

---

## 2. `ivn/`: Survival Analysis dataset (three vehicles)

**Source:** Hacking and Countermeasure Research Lab (HCRL), Korea University.
<https://ocslab.hksecurity.net/Datasets/survival-ids>

> M. L. Han, B. I. Kwak, H. K. Kim, "Anomaly intrusion detection method for vehicular networks based on survival analysis," *Vehicular Communications*, vol. 14, 2018.

CAN bus traffic from three vehicle models (**Hyundai Sonata**, **Kia Soul**, **Chevrolet Spark**), each with one normal capture and three attack captures.

**Format (CSV, with header):**

```
Timestamp,Arbitration_ID,DLC,Data,Class
1559011998.700385,165,8,17 67 37 BA 99 F0 6E 61,R
```

| Field | Meaning |
|---|---|
| Timestamp | Unix time in seconds |
| Arbitration_ID | CAN ID in hex |
| DLC | Number of data bytes |
| Data | Payload bytes, space-separated hex |
| Class | `R` = normal, `T` = attack |

| Attack | Description |
|---|---|
| Flooding | Bus flooded with a high-priority ID, delaying legitimate messages |
| Fuzzy | Random IDs and payloads injected |
| Malfunction | Messages for a specific ID altered so that an ECU behaves abnormally |

Attack messages (T) per file:

| Vehicle | Flooding | Fuzzy | Malfunction | Normal |
|---|---|---|---|---|
| Sonata | 3,954 / 8,460 | 3,015 / 8,488 | 1,876 / 8,520 | 0 / 8,575 |
| Soul | 3,343 / 8,400 | 2,858 / 8,408 | 1,601 / 8,425 | 0 / 8,447 |
| Spark | 4,322 / 8,489 | 3,643 / 8,515 | 2,377 / 8,564 | 0 / 8,656 |

**Used in:** Experiment 2 (in-vehicle intrusion detection) and Experiment 3 (cross-vehicle generalisation: train on Sonata, test on Soul).

---

## 3. `botiot/`: BoT-IoT dataset

**Source:** UNSW Canberra Cyber Range Lab.
<https://research.unsw.edu.au/projects/bot-iot-dataset>

> N. Koroniotis, N. Moustafa, E. Sitnikova, B. Turnbull, "Towards the development of realistic botnet dataset in the Internet of Things for network forensic analytics: Bot-IoT dataset," *Future Generation Computer Systems*, vol. 100, 2019.

Raw packet captures from an IoT testbed with normal traffic and botnet attacks.

**Format:** `.pcap` (Ethernet, IPv4 TCP/UDP/ICMP). The **folder name is the class label**.

| Class | Description |
|---|---|
| Normal | Benign IoT traffic |
| DoS | Denial of service from a single source (TCP/UDP/HTTP) |
| DDoS | Distributed denial of service from multiple bots |
| Reconnaissance | Port scanning and OS fingerprinting |
| Theft | Keylogging and data exfiltration |

**Used in:** Experiment 1 (extra-vehicle intrusion detection).

---

## 4. `cicids/`: CICIDS2017 dataset

**Source:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick.
<https://www.unb.ca/cic/datasets/ids-2017.html>

> I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization," *ICISSP*, 2018.

Packet captures of a simulated enterprise network over five working days, with labelled flows.

| File | Contents |
|---|---|
| `Monday-WorkingHours.pcap` | Benign traffic only. Used for unlabelled pre-training. |
| `AttackDay-WorkingHours.pcap` | Benign and attack traffic |
| `AttackDay-WorkingHours.pcap_ISCX.csv` | Flow labels for the attack-day capture (from the *GeneratedLabelledFlows* CSVs) |

**Label CSV columns:** `Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol, Timestamp, Label`

Packets in the PCAP are labelled by matching their 5-tuple (src IP, dst IP, src port, dst port, protocol) to a flow in the CSV.

| Label | Flows |
|---|---|
| BENIGN | 197 |
| PortScan | 2,151 |
| Bot | 427 |
| Web Attack – Brute Force | 427 |
| DoS GoldenEye | 398 |
| DoS Hulk | 339 |
| DDoS | 313 |
| Web Attack – XSS | 218 |
| DoS Slowhttptest | 175 |
| DoS slowloris | 175 |
| FTP-Patator | 171 |
| SSH-Patator | 171 |

**Used in:** pre-training (Monday capture) and Experiment 1 (extra-vehicle intrusion detection).

---

## How the data is used

`semantic_extractor.py` turns each dataset into byte sentences:
- **BSL** (byte sentence list), unlabelled, for pre-training (`pretrain.py`)
- **USL** (`label \t group \t sentence`), labelled, for fine-tuning (`finetune.py`)

PCAP files are read with `pcap_reader.py`. CAN files are parsed as CSV or text. If a dataset folder is missing, `run_pipeline.py` skips it along with its experiment.
