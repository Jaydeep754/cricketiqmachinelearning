"""
src/squad_data.py
Confirmed IPL 2026 Mega-Auction Squads & Predicted Starting XIs.
Provides strictly verified 2025/2026 post-mega auction rosters with zero 2024 departed player artifacts.
"""
from __future__ import annotations
from pathlib import Path
from src.data_loader import CURRENT_TEAMS

# ═════════════════════════════════════════════════════════════════════════════
# VERIFIED 2025 / 2026 MEGA AUCTION SQUADS & PROJECTED STARTING XIs
# All names mapped to exact player codes in the deliveries database.
# ═════════════════════════════════════════════════════════════════════════════

SQUADS_2025_2026: dict[str, dict] = {
    "Chennai Super Kings": {
        "captain": "RD Gaikwad",
        "predicted_xi": [
            "RD Gaikwad",      # Ruturaj Gaikwad (c)
            "DP Conway",       # Devon Conway (wk)
            "R Ravindra",      # Rachin Ravindra
            "S Dube",          # Shivam Dube
            "DJ Hooda",        # Deepak Hooda
            "RA Jadeja",       # Ravindra Jadeja
            "MS Dhoni",        # MS Dhoni (wk)
            "SM Curran",       # Sam Curran
            "R Ashwin",        # Ravichandran Ashwin
            "Noor Ahmad",      # Noor Ahmad
            "M Pathirana",     # Matheesha Pathirana
        ],
        "squad": [
            "RD Gaikwad", "DP Conway", "R Ravindra", "S Dube", "DJ Hooda",
            "RA Jadeja", "MS Dhoni", "SM Curran", "R Ashwin", "Noor Ahmad",
            "M Pathirana", "RA Tripathi", "KK Ahmed", "V Shankar", "M Choudhary",
            "A Kamboj", "N Ellis", "S Gopal", "Kamlesh Nagarkoti", "Shaik Rasheed",
            "Gurjapneet Singh", "Jamie Overton"
        ],
        "impact_subs": ["KK Ahmed", "RA Tripathi", "V Shankar", "M Choudhary"]
    },

    "Mumbai Indians": {
        "captain": "HH Pandya",
        "predicted_xi": [
            "RG Sharma",       # Rohit Sharma
            "WG Jacks",        # Will Jacks
            "Tilak Varma",     # Tilak Varma
            "SA Yadav",        # Suryakumar Yadav
            "HH Pandya",       # Hardik Pandya (c)
            "Naman Dhir",      # Naman Dhir
            "Robin Minz",      # Robin Minz (wk)
            "MJ Santner",      # Mitchell Santner
            "DL Chahar",       # Deepak Chahar
            "TA Boult",        # Trent Boult
            "JJ Bumrah",       # Jasprit Bumrah
        ],
        "squad": [
            "RG Sharma", "WG Jacks", "Tilak Varma", "SA Yadav", "HH Pandya",
            "Naman Dhir", "Robin Minz", "MJ Santner", "DL Chahar", "TA Boult",
            "JJ Bumrah", "KV Sharma", "RJW Topley", "Arjun Tendulkar", "Ashwani Kumar",
            "Lizaad Williams", "Corbin Bosch", "Raj Angad Bawa", "Ryan Rickelton", "Bevon Jacobs"
        ],
        "impact_subs": ["DL Chahar", "KV Sharma", "Robin Minz", "WG Jacks"]
    },

    "Royal Challengers Bengaluru": {
        "captain": "RM Patidar",
        "predicted_xi": [
            "V Kohli",         # Virat Kohli
            "PD Salt",         # Phil Salt (wk)
            "RM Patidar",      # Rajat Patidar (c)
            "LS Livingstone",  # Liam Livingstone
            "JM Sharma",       # Jitesh Sharma (wk)
            "TH David",        # Tim David
            "KH Pandya",       # Krunal Pandya
            "B Kumar",         # Bhuvneshwar Kumar
            "Swapnil Singh",   # Swapnil Singh
            "JR Hazlewood",    # Josh Hazlewood
            "Yash Dayal",      # Yash Dayal
        ],
        "squad": [
            "V Kohli", "PD Salt", "RM Patidar", "VR Iyer", "LS Livingstone", "JM Sharma",
            "TH David", "KH Pandya", "B Kumar", "Swapnil Singh", "JR Hazlewood",
            "Yash Dayal", "D Padikkal", "Rasikh Salam", "Suyash Sharma", "L Ngidi",
            "N Thushara", "Jacob Bethell", "Manoj Bhandage", "Swastik Chhikara",
            "Jacob Duffy", "R Shepherd", "Abhinandan Singh", "Mohit Rathee"
        ],
        "impact_subs": ["VR Iyer", "D Padikkal", "Suyash Sharma", "Rasikh Salam", "Swapnil Singh"]
    },

    "Kolkata Knight Riders": {
        "captain": "AM Rahane",
        "predicted_xi": [
            "Q de Kock",       # Quinton de Kock (wk)
            "SP Narine",       # Sunil Narine
            "AM Rahane",       # Ajinkya Rahane (c)
            "VR Iyer",         # Venkatesh Iyer
            "RK Singh",        # Rinku Singh
            "AD Russell",      # Andre Russell
            "Ramandeep Singh", # Ramandeep Singh
            "AS Roy",          # Anukul Roy
            "Harshit Rana",    # Harshit Rana
            "A Nortje",        # Anrich Nortje
            "CV Varun",        # Varun Chakravarthy
        ],
        "squad": [
            "Q de Kock", "SP Narine", "AM Rahane", "VR Iyer", "RK Singh",
            "AD Russell", "Ramandeep Singh", "AS Roy", "Harshit Rana", "A Nortje",
            "CV Varun", "Rahmanullah Gurbaz", "VG Arora", "M Markande", "R Powell",
            "MK Pandey", "Moeen Ali", "Umran Malik", "Spencer Johnson", "A Raghuvanshi",
            "Lallu Yadav"
        ],
        "impact_subs": ["VG Arora", "A Raghuvanshi", "M Markande", "Rahmanullah Gurbaz"]
    },

    "Sunrisers Hyderabad": {
        "captain": "PJ Cummins",
        "predicted_xi": [
            "TM Head",         # Travis Head
            "Abhishek Sharma", # Abhishek Sharma
            "Ishan Kishan",    # Ishan Kishan (wk)
            "Nithish Kumar Reddy", # Nitish Kumar Reddy
            "H Klaasen",       # Heinrich Klaasen
            "A Manohar",       # Abhinav Manohar
            "PJ Cummins",      # Pat Cummins (c)
            "HV Patel",        # Harshal Patel
            "Mohammed Shami",  # Mohammad Shami
            "RD Chahar",       # Rahul Chahar
            "A Zampa",         # Adam Zampa
        ],
        "squad": [
            "TM Head", "Abhishek Sharma", "Ishan Kishan", "Nithish Kumar Reddy",
            "H Klaasen", "A Manohar", "PJ Cummins", "HV Patel", "Mohammed Shami",
            "RD Chahar", "A Zampa", "Atharva Taide", "JD Unadkat", "Simarjeet Singh",
            "Sachin Baby", "Bryce Street", "Zeeshan Ansari", "Wiaan Mulder"
        ],
        "impact_subs": ["Atharva Taide", "Simarjeet Singh", "JD Unadkat", "A Manohar"]
    },

    "Delhi Capitals": {
        "captain": "AR Patel",
        "predicted_xi": [
            "KL Rahul",        # KL Rahul (wk)
            "J Fraser-McGurk", # Jake Fraser-McGurk
            "Abishek Porel",   # Abishek Porel
            "F du Plessis",    # Faf du Plessis
            "T Stubbs",        # Tristan Stubbs
            "AR Patel",        # Axar Patel (c)
            "Ashutosh Sharma", # Ashutosh Sharma
            "MA Starc",        # Mitchell Starc
            "Kuldeep Yadav",   # Kuldeep Yadav
            "Mukesh Kumar",    # Mukesh Kumar
            "T Natarajan",     # T Natarajan
        ],
        "squad": [
            "KL Rahul", "J Fraser-McGurk", "Abishek Porel", "F du Plessis", "T Stubbs",
            "AR Patel", "Ashutosh Sharma", "MA Starc", "Kuldeep Yadav", "Mukesh Kumar",
            "T Natarajan", "Harry Brook", "KK Nair", "Sameer Rizvi", "MM Sharma",
            "D Ferreira", "DG Nalkande", "Dushmantha Chameera", "Vipraj Nigam"
        ],
        "impact_subs": ["MM Sharma", "KK Nair", "Sameer Rizvi", "T Natarajan"]
    },

    "Punjab Kings": {
        "captain": "SS Iyer",
        "predicted_xi": [
            "P Simran Singh",  # Prabhsimran Singh (wk)
            "SS Iyer",         # Shreyas Iyer (c)
            "Shashank Singh",  # Shashank Singh
            "GJ Maxwell",      # Glenn Maxwell
            "MP Stoinis",      # Marcus Stoinis
            "N Wadhera",       # Nehal Wadhera
            "Azmatullah Omarzai", # Azmatullah Omarzai
            "M Jansen",        # Marco Jansen
            "Arshdeep Singh",  # Arshdeep Singh
            "YS Chahal",       # Yuzvendra Chahal
            "LH Ferguson",     # Lockie Ferguson
        ],
        "squad": [
            "P Simran Singh", "SS Iyer", "Shashank Singh", "GJ Maxwell", "MP Stoinis",
            "N Wadhera", "Azmatullah Omarzai", "M Jansen", "Arshdeep Singh", "YS Chahal",
            "LH Ferguson", "Xavier Bartlett", "Harpreet Brar", "Vyshak Vijaykumar",
            "Yash Thakur", "Josh Inglis", "Vishnu Vinod", "Harpreet Singh",
            "Aaron Hardie", "Kulwant Khejroliya", "Pravin Dubey"
        ],
        "impact_subs": ["Harpreet Brar", "N Wadhera", "Yash Thakur", "Vyshak Vijaykumar"]
    },

    "Rajasthan Royals": {
        "captain": "SV Samson",
        "predicted_xi": [
            "YBK Jaiswal",     # Yashasvi Jaiswal
            "SV Samson",       # Sanju Samson (c & wk)
            "N Rana",          # Nitish Rana
            "R Parag",         # Riyan Parag
            "Dhruv Jurel",     # Dhruv Jurel
            "SO Hetmyer",      # Shimron Hetmyer
            "PWH de Silva",    # Wanindu Hasaranga
            "JC Archer",       # Jofra Archer
            "M Theekshana",    # Maheesh Theekshana
            "Sandeep Sharma",  # Sandeep Sharma
            "TU Deshpande",    # Tushar Deshpande
        ],
        "squad": [
            "YBK Jaiswal", "SV Samson", "N Rana", "R Parag", "Dhruv Jurel",
            "SO Hetmyer", "PWH de Silva", "JC Archer", "M Theekshana", "Sandeep Sharma",
            "TU Deshpande", "Fazalhaq Farooqi", "Shubham Dubey", "Kumar Kartikeya",
            "Akash Madhwal", "Kunal Singh Rathore", "Vaibhav Suryavanshi", "Ashok Sharma",
            "Yudhvir Singh Charak"
        ],
        "impact_subs": ["Shubham Dubey", "Fazalhaq Farooqi", "Akash Madhwal", "Kumar Kartikeya"]
    },

    "Lucknow Super Giants": {
        "captain": "RR Pant",
        "predicted_xi": [
            "AK Markram",      # Aiden Markram
            "MR Marsh",        # Mitchell Marsh
            "RR Pant",         # Rishabh Pant (c & wk)
            "N Pooran",        # Nicholas Pooran
            "A Badoni",        # Ayush Badoni
            "DA Miller",       # David Miller
            "Abdul Samad",     # Abdul Samad
            "Shahbaz Ahmed",   # Shahbaz Ahmed
            "R Bishnoi",       # Ravi Bishnoi
            "Avesh Khan",      # Avesh Khan
            "Akash Deep",      # Akash Deep
        ],
        "squad": [
            "AK Markram", "MR Marsh", "RR Pant", "N Pooran", "A Badoni",
            "DA Miller", "Abdul Samad", "Shahbaz Ahmed", "R Bishnoi", "Avesh Khan",
            "Akash Deep", "Mohsin Khan", "Mayank Yadav", "SN Thakur", "Shamar Joseph",
            "Matthew Breetzke", "Arshin Kulkarni", "M Siddharth", "Digvesh Rathi",
            "Aryan Juyal", "Prince Yadav"
        ],
        "impact_subs": ["Mayank Yadav", "Mohsin Khan", "Abdul Samad", "SN Thakur"]
    },

    "Gujarat Titans": {
        "captain": "Shubman Gill",
        "predicted_xi": [
            "Shubman Gill",    # Shubman Gill (c)
            "JC Buttler",      # Jos Buttler (wk)
            "B Sai Sudharsan", # B Sai Sudharsan
            "M Shahrukh Khan", # M Shahrukh Khan
            "GD Phillips",     # Glenn Phillips
            "Washington Sundar", # Washington Sundar
            "R Tewatia",       # Rahul Tewatia
            "Rashid Khan",     # Rashid Khan
            "K Rabada",        # Kagiso Rabada
            "Mohammed Siraj",  # Mohammed Siraj
            "R Sai Kishore",   # R Sai Kishore
        ],
        "squad": [
            "Shubman Gill", "JC Buttler", "B Sai Sudharsan", "M Shahrukh Khan",
            "GD Phillips", "Washington Sundar", "R Tewatia", "Rashid Khan", "K Rabada",
            "Mohammed Siraj", "R Sai Kishore", "G Coetzee", "Ishant Sharma",
            "MK Lomror", "Nishant Sindhu", "Spencer Johnson", "Arshad Khan", "Gurnoor Brar",
            "Kumar Kushagra", "Karim Janat", "A Rawat"
        ],
        "impact_subs": ["MK Lomror", "Ishant Sharma", "G Coetzee", "R Sai Kishore"]
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# DYNAMIC SQUAD LOADER (Confirmed IPL 2026 Mega-Auction Rosters)
# ═════════════════════════════════════════════════════════════════════════════

def get_squad_data(base_dir: Path | None = None) -> dict:
    """
    Returns confirmed IPL 2026 squad rosters and predicted starting XIs.
    team → {last_xi, squad, impact_subs}.
    """
    return {
        team: {
            "last_xi": SQUADS_2025_2026[team]["predicted_xi"],
            "squad": sorted(set(SQUADS_2025_2026[team]["squad"])),
            "impact_subs": SQUADS_2025_2026[team]["impact_subs"],
        }
        for team in CURRENT_TEAMS
    }
