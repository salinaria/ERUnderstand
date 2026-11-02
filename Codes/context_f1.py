import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# ----------------- Context Mapping ----------------- #
CONTEXT_MAPPING = {
    # Sports & Athletics (20 contexts)
    "Footballmatch": "Sports & Athletics",
    "Basketballgame": "Sports & Athletics",
    "Volleyballtournament": "Sports & Athletics",
    "Tennismatch": "Sports & Athletics",
    "Swimmingcompetition": "Sports & Athletics",
    "Gymworkout": "Sports & Athletics",
    "Runningmarathon": "Sports & Athletics",
    "Skiingevent": "Sports & Athletics",
    "Boxingring": "Sports & Athletics",
    "Cricketpractice": "Sports & Athletics",
    "Baseballgame": "Sports & Athletics",
    "Golfcourse": "Sports & Athletics",
    "Surfingsession": "Sports & Athletics",
    "Rugbymatch": "Sports & Athletics",
    "Tabletennisgame": "Sports & Athletics",
    "Badmintoncompetition": "Sports & Athletics",
    "Iceskatingrink": "Sports & Athletics",
    "Rockclimbing": "Sports & Athletics",
    "Karatedojo": "Sports & Athletics",
    "Fencingmatch": "Sports & Athletics",
    
    # Home & Household Activities (20 contexts)
    "Cookinginthekitchen": "Home & Household Activities",
    "WatchingTVinthelivingroom": "Home & Household Activities",
    "Sleepinginthebedroom": "Home & Household Activities",
    "Cleaningthehouse": "Home & Household Activities",
    "Gardeninginthebackyard": "Home & Household Activities",
    "Doinglaundry": "Home & Household Activities",
    "Decoratingforaholiday": "Home & Household Activities",
    "Hostingadinnerparty": "Home & Household Activities",
    "Homerenovation": "Home & Household Activities",
    "Playingwithpets": "Home & Household Activities",
    "Familydinner": "Home & Household Activities",
    "Changingalightbulb": "Home & Household Activities",
    "Fixingasink": "Home & Household Activities",
    "Homeofficework": "Home & Household Activities",
    "Babysitting": "Home & Household Activities",
    "Movingfurniture": "Home & Household Activities",
    "Readinginbed": "Home & Household Activities",
    "Organizingcloset": "Home & Household Activities",
    "Vacuumingcarpet": "Home & Household Activities",
    "Wateringplants": "Home & Household Activities",
    
    # Social Events & Gatherings (20 contexts)
    "Birthdayparty": "Social Events & Gatherings",
    "Weddingreception": "Social Events & Gatherings",
    "Funeralceremony": "Social Events & Gatherings",
    "Familyreunion": "Social Events & Gatherings",
    "Schoolreunion": "Social Events & Gatherings",
    "Communityfestival": "Social Events & Gatherings",
    "Firstdate": "Social Events & Gatherings",
    "Breakupconversation": "Social Events & Gatherings",
    "Religiousgathering": "Social Events & Gatherings",
    "Onlinegamingchat": "Social Events & Gatherings",
    "Picnicwithfriends": "Social Events & Gatherings",
    "Barbecueparty": "Social Events & Gatherings",
    "Clubnight": "Social Events & Gatherings",
    "Bookclub": "Social Events & Gatherings",
    "Charityevent": "Social Events & Gatherings",
    "Concertnight": "Social Events & Gatherings",
    "Neighborhoodwatchmeeting": "Social Events & Gatherings",
    "Speeddatingevent": "Social Events & Gatherings",
    "Grouphike": "Social Events & Gatherings",
    "Dinnerwithin-laws": "Social Events & Gatherings",
    
    # Education & Learning (20 contexts)
    "Classroomlecture": "Education & Learning",
    "Takinganexam": "Education & Learning",
    "Groupprojectmeeting": "Education & Learning",
    "Sciencelabexperiment": "Education & Learning",
    "Schoolfieldtrip": "Education & Learning",
    "Onlinecoursesession": "Education & Learning",
    "Studyinginalibrary": "Education & Learning",
    "Graduationceremony": "Education & Learning",
    "Teacher-parentmeeting": "Education & Learning",
    "Tutoringsession": "Education & Learning",
    "Thesisdefense": "Education & Learning",
    "Collegefair": "Education & Learning",
    "Codingbootcamp": "Education & Learning",
    "Languageclass": "Education & Learning",
    "Schoolassembly": "Education & Learning",
    "Disciplinarymeeting": "Education & Learning",
    "Artclass": "Education & Learning",
    "Musicrehearsal": "Education & Learning",
    "Historypresentation": "Education & Learning",
    "Sciencefair": "Education & Learning",
    
    # Work & Professional (20 contexts)
    "Jobinterview": "Work & Professional",
    "Zoommeeting": "Work & Professional",
    "Businesspresentation": "Work & Professional",
    "Teambrainstorming": "Work & Professional",
    "Coffeebreak": "Work & Professional",
    "Codereview": "Work & Professional",
    "Writingareport": "Work & Professional",
    "Salespitch": "Work & Professional",
    "Corporatetraining": "Work & Professional",
    "Officeparty": "Work & Professional",
    "Clientonboarding": "Work & Professional",
    "Performancereview": "Work & Professional",
    "Dailystandup": "Work & Professional",
    "Remoteworksetup": "Work & Professional",
    "Workingovertime": "Work & Professional",
    "Filingdocuments": "Work & Professional",
    "Conflictresolution": "Work & Professional",
    "Team-buildingevent": "Work & Professional",
    "HRorientation": "Work & Professional",
    "Workplacesafetydrill": "Work & Professional",
    
    # Healthcare & Medical (20 contexts)
    "Doctorappointment": "Healthcare & Medical",
    "Hospitalsurgery": "Healthcare & Medical",
    "Mentalhealththerapy": "Healthcare & Medical",
    "Dentalcleaning": "Healthcare & Medical",
    "Pharmacyvisit": "Healthcare & Medical",
    "Emergencyroomtreatment": "Healthcare & Medical",
    "Healthcheckup": "Healthcare & Medical",
    "Physicaltherapysession": "Healthcare & Medical",
    "Vaccinationclinic": "Healthcare & Medical",
    "Nutritionconsultation": "Healthcare & Medical",
    "Chiropracticsession": "Healthcare & Medical",
    "Eyeexam": "Healthcare & Medical",
    "Hearingtest": "Healthcare & Medical",
    "Blooddonation": "Healthcare & Medical",
    "CPRtraining": "Healthcare & Medical",
    "Prenatalcheckup": "Healthcare & Medical",
    "Childbirthdelivery": "Healthcare & Medical",
    "Medicalimaging": "Healthcare & Medical",
    "Prescriptionrefill": "Healthcare & Medical",
    "Firstaidscenario": "Healthcare & Medical",
    
    # Legal & Government Services (20 contexts)
    "Courtroomtrial": "Legal & Government Services",
    "Policestationreport": "Legal & Government Services",
    "Passportapplication": "Legal & Government Services",
    "Votingstation": "Legal & Government Services",
    "Legalconsultation": "Legal & Government Services",
    "Immigrationinterview": "Legal & Government Services",
    "Notarysigning": "Legal & Government Services",
    "Taxofficevisit": "Legal & Government Services",
    "Juryduty": "Legal & Government Services",
    "Citycouncilmeeting": "Legal & Government Services",
    "Arrestprocedure": "Legal & Government Services",
    "Licenserenewal": "Legal & Government Services",
    "Divorcehearing": "Legal & Government Services",
    "Publichearing": "Legal & Government Services",
    "Bailsetting": "Legal & Government Services",
    "Legaldeposition": "Legal & Government Services",
    "Smallclaimscourt": "Legal & Government Services",
    "Filingalawsuit": "Legal & Government Services",
    "Prisonvisit": "Legal & Government Services",
    "Legalaidclinic": "Legal & Government Services",
    
    # Travel & Transportation (20 contexts)
    "Airportsecuritycheck": "Travel & Transportation",
    "Boardinganairplane": "Travel & Transportation",
    "Hotelcheck-in": "Travel & Transportation",
    "Roadtrip": "Travel & Transportation",
    "Trainstationwaiting": "Travel & Transportation",
    "Subwaycommute": "Travel & Transportation",
    "Passportcontrol": "Travel & Transportation",
    "Busterminal": "Travel & Transportation",
    "Beachvacation": "Travel & Transportation",
    "Backpackingabroad": "Travel & Transportation",
    "Travelitineraryplanning": "Travel & Transportation",
    "Carrentalpickup": "Travel & Transportation",
    "Cruiseshipboarding": "Travel & Transportation",
    "Touristsightseeing": "Travel & Transportation",
    "Culturalexchange": "Travel & Transportation",
    "Mountaintrekking": "Travel & Transportation",
    "Desertsafari": "Travel & Transportation",
    "Visaapplication": "Travel & Transportation",
    "Lostluggageclaim": "Travel & Transportation",
    "Souvenirshopping": "Travel & Transportation",
    
    # Shopping & Retail (20 contexts)
    "Groceryshopping": "Shopping & Retail",
    "BlackFridaysale": "Shopping & Retail",
    "Onlinecheckout": "Shopping & Retail",
    "Return/refundcounter": "Shopping & Retail",
    "Farmer'smarket": "Shopping & Retail",
    "Thriftstorevisit": "Shopping & Retail",
    "Hairsalonappointment": "Shopping & Retail",
    "Buyingacar": "Shopping & Retail",
    "Mallfoodcourt": "Shopping & Retail",
    "Techstoredemo": "Shopping & Retail",
    "Boutiquevisit": "Shopping & Retail",
    "Petstoretrip": "Shopping & Retail",
    "Buyingschoolsupplies": "Shopping & Retail",
    "Homeapplianceshopping": "Shopping & Retail",
    "Jewelrystorebrowsing": "Shopping & Retail",
    "Toystore": "Shopping & Retail",
    "Tryingclothesinfittingroom": "Shopping & Retail",
    "Windowshopping": "Shopping & Retail",
    "Payingwithcoupons": "Shopping & Retail",
    "Askingforadiscount": "Shopping & Retail",
    
    # Technology & Digital Activities (20 contexts)
    "Writingcode": "Technology & Digital Activities",
    "Videocall": "Technology & Digital Activities",
    "Socialmediapost": "Technology & Digital Activities",
    "Onlinebanking": "Technology & Digital Activities",
    "Cybersecuritybreach": "Technology & Digital Activities",
    "WatchingYouTube": "Technology & Digital Activities",
    "PlayingVRgames": "Technology & Digital Activities",
    "AIchatbotconversation": "Technology & Digital Activities",
    "Installinganapp": "Technology & Digital Activities",
    "Usingasmartassistant": "Technology & Digital Activities",
    "BrowsingReddit": "Technology & Digital Activities",
    "Updatingsoftware": "Technology & Digital Activities",
    "Editingavideo": "Technology & Digital Activities",
    "Settingupanetwork": "Technology & Digital Activities",
    "Creatingawebsite": "Technology & Digital Activities",
    "Streamingagame": "Technology & Digital Activities",
    "Managingadatabase": "Technology & Digital Activities",
    "Fixingaprinter": "Technology & Digital Activities",
    "Usinga3Dprinter": "Technology & Digital Activities",
    "LearningwithAItutor": "Technology & Digital Activities",
}

# Error categories for analysis
ERROR_CATEGORIES = [
    "Ent-extra", "Ent-missed",
    "Attr-missed", "Attr-extra", "Attr-miscon",
    "multival-missed", "derived-missed",
    "comp-extra", "comp-missed",
    "PrimKey-extra", "PrimKey-missed",
    "Rel-extra", "Rel-missed", "Rel-miscon", "Cardinality",
    "WeakRel-extra", "WeakRel-missed",
    "WeakEnt-extra", "WeakEnt-missed",
    "IsA-missed", "IsA-extra", "IsA-miscon", "N-ary-missed"
]

CONTEXTS = [
    "Sports & Athletics",
    "Home & Household Activities",
    "Social Events & Gatherings",
    "Education & Learning",
    "Work & Professional",
    "Healthcare & Medical",
    "Legal & Government Services",
    "Travel & Transportation",
    "Shopping & Retail",
    "Technology & Digital Activities"
]

# ----------------- F1 Computation ----------------- #
def compute_f1(gt: int, extra: int, missed: int, miscon: int = 0) -> float:
    """
    Compute F1 score for a given category.
    Misconnections are treated as 0.5 FP and 0.5 FN (fractional).
    """
    f = 0.5  # fraction for misconnections
    if gt <= 0:
        return 0.0

    TP = max(0, gt - missed)
    FP = extra + (1 - f) * miscon
    FN = missed + f * miscon

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / gt if gt > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return round(2 * precision * recall / (precision + recall), 3)


# ----------------- Analysis Function ----------------- #
def analyze_contexts(error_csv_path: str, gt_csv_path: str, output_path: str = None):
    """
    Analyze errors by context category and compute F1 scores.
    
    Args:
        error_csv_path: Path to the error CSV file
        gt_csv_path: Path to ground truth statistics CSV
        output_path: Optional path to save the results
    """
    # Load data
    df_errors = pd.read_csv(error_csv_path)
    df_gt = pd.read_csv(gt_csv_path)
    
    # Remove summary rows if they exist
    df_errors = df_errors[~df_errors['Diagram'].str.contains('Total|Mean|Sum', case=False, na=False)]
    
    # Add context column to errors
    df_errors['Context'] = df_errors['Diagram'].map(CONTEXT_MAPPING)
    
    # Handle unmapped contexts
    unmapped = df_errors[df_errors['Context'].isna()]
    if not unmapped.empty:
        print(f"Warning: {len(unmapped)} diagrams not mapped to contexts:")
        print(unmapped['Diagram'].tolist())
    
    # Filter out unmapped
    df_errors = df_errors[df_errors['Context'].notna()]
    
    # Merge with ground truth
    df_merged = pd.merge(
        df_errors, 
        df_gt, 
        left_on='Diagram', 
        right_on='Diagram Number',
        how='left'
    )
    
    # Initialize results dictionary
    results = []
    
    # Analyze each context
    for context in CONTEXTS:
        context_data = df_merged[df_merged['Context'] == context]
        
        if context_data.empty:
            print(f"Warning: No data for context '{context}'")
            continue
        
        # Calculate total errors for each category
        error_totals = {}
        for col in ERROR_CATEGORIES:
            error_totals[col] = context_data[col].sum()
        
        # Calculate ground truth totals
        gt_totals = {
            'GT_Entities': context_data['entities'].sum(),
            'GT_RegAttr': context_data['attributes'].sum(),
            'GT_Multivalued': context_data['multivalued'].sum(),
            'GT_Derived': context_data['derived'].sum(),
            'GT_Composite': context_data['composite'].sum(),
            'GT_PrimaryKeys': context_data['primary_keys'].sum(),
            'GT_Relationships': context_data['relationships'].sum(),
            'GT_Cardinality': context_data['relationships'].sum(),
            'GT_WeakEntities': context_data['weak_entities'].sum(),
            'GT_WeakRelationships': context_data['weak_relationships'].sum(),
            'GT_IsA': context_data['IsA'].sum(),
            'GT_Nary': context_data['n_ary'].sum(),
        }
        
        # Calculate F1 scores
        f1_scores = {
            'F1_Entities': compute_f1(
                gt_totals['GT_Entities'], 
                error_totals['Ent-extra'], 
                error_totals['Ent-missed'], 
                0
            ),
            'F1_RegAttr': compute_f1(
                gt_totals['GT_RegAttr'],
                error_totals['Attr-extra'],
                error_totals['Attr-missed'],
                error_totals['Attr-miscon']
            ),
            'F1_Multivalued': compute_f1(
                gt_totals['GT_Multivalued'],
                0,
                error_totals['multival-missed'],
                0
            ),
            'F1_Derived': compute_f1(
                gt_totals['GT_Derived'],
                0,
                error_totals['derived-missed'],
                0
            ),
            'F1_Composite': compute_f1(
                gt_totals['GT_Composite'],
                0,
                error_totals['comp-missed'],
                0
            ),
            'F1_PrimaryKeys': compute_f1(
                gt_totals['GT_PrimaryKeys'],
                error_totals['PrimKey-extra'],
                error_totals['PrimKey-missed'],
                0
            ),
            'F1_Relationships': compute_f1(
                gt_totals['GT_Relationships'],
                error_totals['Rel-extra'],
                error_totals['Rel-missed'],
                error_totals['Rel-miscon']
            ),
            'F1_Cardinality': compute_f1(
                gt_totals['GT_Cardinality'],
                0,
                error_totals['Cardinality'],
                0
            ),
            'F1_WeakEntities': compute_f1(
                gt_totals['GT_WeakEntities'],
                error_totals['WeakEnt-extra'],
                error_totals['WeakEnt-missed'],
                0
            ),
            'F1_WeakRelationships': compute_f1(
                gt_totals['GT_WeakRelationships'],
                error_totals['WeakRel-extra'],
                error_totals['WeakRel-missed'],
                0
            ),
            'F1_IsA': compute_f1(
                gt_totals['GT_IsA'],
                error_totals['IsA-extra'],
                error_totals['IsA-missed'],
                error_totals['IsA-miscon']
            ),
            'F1_Nary': compute_f1(
                gt_totals['GT_Nary'],
                0,
                error_totals['N-ary-missed'],
                0
            ),
        }
        
        # Calculate aggregate metrics
        total_errors = sum(error_totals.values())
        total_gt = context_data['total'].sum()
        num_diagrams = len(context_data)
        
        # Macro F1: Simple average of all F1 scores
        valid_f1s = [f1 for f1 in f1_scores.values() if f1 > 0]
        macro_f1 = round(np.mean(valid_f1s), 3) if valid_f1s else 0.0
        
        # Weighted F1: Weighted by ground truth frequency
        gt_values = np.array(list(gt_totals.values()))
        f1_values = np.array(list(f1_scores.values()))
        non_zero_mask = gt_values > 0
        
        if non_zero_mask.any():
            weighted_f1 = round(
                np.average(f1_values[non_zero_mask], weights=gt_values[non_zero_mask]),
                3
            )
        else:
            weighted_f1 = 0.0
        
        # Balanced F1: Mix of equal and frequency-based weights (balance_factor = 0.3)
        if non_zero_mask.any():
            gt_nonzero = gt_values[non_zero_mask]
            f1_nonzero = f1_values[non_zero_mask]
            
            freq_weights = gt_nonzero / gt_nonzero.sum()
            equal_weights = np.ones_like(freq_weights) / len(freq_weights)
            balanced_weights = 0.3 * freq_weights + 0.7 * equal_weights
            
            balanced_f1 = round(np.average(f1_nonzero, weights=balanced_weights), 3)
        else:
            balanced_f1 = 0.0
        
        # Build result row
        result = {
            'Context': context,
            'Num_Diagrams': num_diagrams,
            'Total_Errors': total_errors,
            'Total_GT': int(total_gt),
            **error_totals,
            **gt_totals,
            **f1_scores,
            'Macro_F1': macro_f1,
            'Balanced_F1': balanced_f1,
            'Weighted_F1': weighted_f1
        }
        
        results.append(result)
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Add overall summary
    overall = {
        'Context': 'OVERALL',
        'Num_Diagrams': df_merged['Diagram'].nunique(),
        'Total_Errors': df_merged['Sum'].sum(),
        'Total_GT': int(df_merged['total'].sum()),
    }
    
    # Overall error totals
    for col in ERROR_CATEGORIES:
        overall[col] = df_merged[col].sum()
    
    # Overall GT totals
    overall['GT_Entities'] = df_merged['entities'].sum()
    overall['GT_RegAttr'] = df_merged['attributes'].sum()
    overall['GT_Multivalued'] = df_merged['multivalued'].sum()
    overall['GT_Derived'] = df_merged['derived'].sum()
    overall['GT_Composite'] = df_merged['composite'].sum()
    overall['GT_PrimaryKeys'] = df_merged['primary_keys'].sum()
    overall['GT_Relationships'] = df_merged['relationships'].sum()
    overall['GT_Cardinality'] = df_merged['relationships'].sum()
    overall['GT_WeakEntities'] = df_merged['weak_entities'].sum()
    overall['GT_WeakRelationships'] = df_merged['weak_relationships'].sum()
    overall['GT_IsA'] = df_merged['IsA'].sum()
    overall['GT_Nary'] = df_merged['n_ary'].sum()
    
    # Overall F1 scores
    overall['F1_Entities'] = compute_f1(overall['GT_Entities'], overall['Ent-extra'], overall['Ent-missed'], 0)
    overall['F1_RegAttr'] = compute_f1(overall['GT_RegAttr'], overall['Attr-extra'], overall['Attr-missed'], overall['Attr-miscon'])
    overall['F1_Multivalued'] = compute_f1(overall['GT_Multivalued'], 0, overall['multival-missed'], 0)
    overall['F1_Derived'] = compute_f1(overall['GT_Derived'], 0, overall['derived-missed'], 0)
    overall['F1_Composite'] = compute_f1(overall['GT_Composite'], 0, overall['comp-missed'], 0)
    overall['F1_PrimaryKeys'] = compute_f1(overall['GT_PrimaryKeys'], overall['PrimKey-extra'], overall['PrimKey-missed'], 0)
    overall['F1_Relationships'] = compute_f1(overall['GT_Relationships'], overall['Rel-extra'], overall['Rel-missed'], overall['Rel-miscon'])
    overall['F1_Cardinality'] = compute_f1(overall['GT_Cardinality'], 0, overall['Cardinality'], 0)
    overall['F1_WeakEntities'] = compute_f1(overall['GT_WeakEntities'], overall['WeakEnt-extra'], overall['WeakEnt-missed'], 0)
    overall['F1_WeakRelationships'] = compute_f1(overall['GT_WeakRelationships'], overall['WeakRel-extra'], overall['WeakRel-missed'], 0)
    overall['F1_IsA'] = compute_f1(overall['GT_IsA'], overall['IsA-extra'], overall['IsA-missed'], overall['IsA-miscon'])
    overall['F1_Nary'] = compute_f1(overall['GT_Nary'], 0, overall['N-ary-missed'], 0)
    
    # Overall aggregate F1 scores
    overall_gt_values = np.array([
        overall['GT_Entities'], overall['GT_RegAttr'], overall['GT_Multivalued'],
        overall['GT_Derived'], overall['GT_Composite'], overall['GT_PrimaryKeys'],
        overall['GT_Relationships'], overall['GT_Cardinality'], overall['GT_WeakEntities'],
        overall['GT_WeakRelationships'], overall['GT_IsA'], overall['GT_Nary']
    ])
    overall_f1_values = np.array([
        overall['F1_Entities'], overall['F1_RegAttr'], overall['F1_Multivalued'],
        overall['F1_Derived'], overall['F1_Composite'], overall['F1_PrimaryKeys'],
        overall['F1_Relationships'], overall['F1_Cardinality'], overall['F1_WeakEntities'],
        overall['F1_WeakRelationships'], overall['F1_IsA'], overall['F1_Nary']
    ])
    overall_non_zero_mask = overall_gt_values > 0
    
    # Macro F1
    overall['Macro_F1'] = round(overall_f1_values[overall_non_zero_mask].mean(), 3) if overall_non_zero_mask.any() else 0.0
    
    # Weighted F1
    if overall_non_zero_mask.any():
        overall['Weighted_F1'] = round(
            np.average(overall_f1_values[overall_non_zero_mask], 
                      weights=overall_gt_values[overall_non_zero_mask]), 3
        )
    else:
        overall['Weighted_F1'] = 0.0
    
    # Balanced F1
    if overall_non_zero_mask.any():
        overall_gt_nonzero = overall_gt_values[overall_non_zero_mask]
        overall_f1_nonzero = overall_f1_values[overall_non_zero_mask]
        
        freq_weights = overall_gt_nonzero / overall_gt_nonzero.sum()
        equal_weights = np.ones_like(freq_weights) / len(freq_weights)
        balanced_weights = 0.3 * freq_weights + 0.7 * equal_weights
        
        overall['Balanced_F1'] = round(np.average(overall_f1_nonzero, weights=balanced_weights), 3)
    else:
        overall['Balanced_F1'] = 0.0
    
    df_results = pd.concat([df_results, pd.DataFrame([overall])], ignore_index=True)
    
    # Save if output path provided
    if output_path:
        df_results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    return df_results


# ----------------- Main Execution ----------------- #
def main():
    """
    Main function to analyze contexts across all models and difficulties.
    Uses the same file structure as the F1 score calculation code.
    """
    models = [
        "Claude-4.6-sonnet", "GPT-5.2", "Gemini-3-pro", "Grok-2-vision",
        "Qwen-3.6", "Llama-3.2", "Gemma-3", "GLM-4.5",
    ]
    difficulties = ["easy", "med", "med-biased", "med-random", 'med-pk', 
                   'med-pk-empty', 'med-persian', 'translated-persian',
                   "hard", "extreme", "IsA", 'Web', 'Web-Silber',
                   'Web-Graphviz', 'Instructional', 'Instructional-Silber', 'Instructional-Graphviz']
    
    # Store all results
    all_results = []
    
    for model in models:
        for difficulty in difficulties:
            try:
                # Construct file paths matching the F1 score code structure
                error_csv = f"Results/F1/{model}/{difficulty}.csv"
                
                # Determine correct ground truth file
                if difficulty in ['med-biased', 'med-random', 'med-pk', 
                                'med-pk-empty', 'med-persian', 'translated-persian']:
                    gt_csv = "Dataset/stats/sorted_stats_med.csv"
                elif difficulty in ['Web-Silber', 'Web-Graphviz', 'Web']:
                    gt_csv = "Dataset/stats/sorted_stats_Web.csv"
                elif difficulty in ['Instructional', 'Instructional-Silber', 'Instructional-Graphviz']:
                    gt_csv = "Dataset/stats/sorted_stats_Instructional.csv"
                else:
                    gt_csv = f"Dataset/stats/sorted_stats_{difficulty}.csv"
                
                # Output path
                output_csv = f"Results/F1/{model}/context_analysis_{difficulty}.csv"
                
                print(f"\nProcessing {model} - {difficulty}...")
                
                # Run analysis
                results = analyze_contexts(error_csv, gt_csv, output_csv)
                
                # Add model and difficulty info
                results['Model'] = model
                results['Difficulty'] = difficulty
                
                # Store for summary
                all_results.append(results)
                
                print(f"✓ Completed {model} - {difficulty}")
                
            except FileNotFoundError as e:
                print(f"✗ Missing files for {model} - {difficulty}: {e}")
            except Exception as e:
                print(f"✗ Error processing {model} - {difficulty}: {e}")
    
    # Create summary across all models and difficulties
    if all_results:
        summary_df = pd.concat(all_results, ignore_index=True)
        summary_output = "Results/F1/context_analysis_summary.csv"
        summary_df.to_csv(summary_output, index=False)
        print(f"\n{'='*80}")
        print(f"Summary of all results saved to: {summary_output}")
        print(f"{'='*80}\n")
        
        # Display overall statistics by context
        print("\nOVERALL STATISTICS BY CONTEXT (Across All Models/Difficulties):")
        print("="*80)
        
        context_summary = summary_df[summary_df['Context'] != 'OVERALL'].groupby('Context').agg({
            'Num_Diagrams': 'sum',
            'Total_Errors': 'sum',
            'Total_GT': 'sum',
            'Macro_F1': 'mean',
            'Balanced_F1': 'mean',
            'Weighted_F1': 'mean'
        }).round(3)
        
        print(context_summary.to_string())
        print("\n")


if __name__ == "__main__":
    main()