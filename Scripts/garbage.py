"""
Here is where I keep my old scripts that are now useless.
"""
import numpy as np
import re


def ComputeKwDistribution(abstract, kw, normalized = True):
    """Function that returns the normalized(optional) distribution of words in a sentence"""
    abstract = abstract.lower()
    kw       = kw.lower()

    split1 = abstract.split(kw)

    positions = []
    phrase_length = 0
    for split_el in split1:
        # Replacing punctuations with spaces so they don't count as words
        clean = re.sub(r"""
               [,.;@#?!&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               split_el, flags=re.VERBOSE)

        temp_split = clean.split()
        phrase_length += len(temp_split)
        positions.append(phrase_length)

        phrase_length+=1 # Takes into account the kw

    positions = positions[:-1]
    phrase_length-=1#Removes the last kw added for nothing

    distribution = np.array([0 for i in range(phrase_length)])
    distribution[np.array(positions).astype(int)] = 1

    return distribution/distribution.sum() if normalized and distribution.sum()!=0 else distribution

def ComputeKwDistribution2(abstract, kw, normalized = True):
    """
        Function that returns the normalized(optional) distribution of words in a sentence.
        This time around we consider each word to be seperate, keywords comprised of multiple words can occur on multiple slots.
    """
    ## Preproc kw and abstract
    abstract = abstract.lower()
    kw       = kw.lower()
    # remove punctuation from asbtract and keywords
    abstract = re.sub(r"""
               [,.;@#?!&$()]+  # Accept one or more copies of punctuation and parenthese
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               abstract, flags=re.VERBOSE)
    kw       = re.sub(r"""
               [,.;@#?!&$()]+  # Accept one or more copies of punctuation and parenthese
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               kw, flags=re.VERBOSE)
    
    ## Split keyword and abstract
    kw_split = kw.split()
    ab_split = np.array(abstract.split())

    ## Find the occurences of the first word of the keyword
    pos_first = np.where(ab_split == kw_split[0])[0]
    positions = []
    for pos in pos_first:
        i = pos
        temp_pos = []
        kw_found = True
        for el in kw_split:
            if(el == ab_split[i]):
                temp_pos.append(i)
            else:
                kw_found = False
                break
                
            i += 1
            
        if(kw_found):
            positions.append(temp_pos)

    phrase_length = len(ab_split)

    distribution = np.array([0 for i in range(phrase_length)])
    distribution[np.array(positions).astype(int)] = 1

    return distribution/distribution.sum() if normalized and distribution.sum()!=0 else distribution


def ExtractKeywordOrderFromAbstract(doi, abstract, ccs, labeled_data_dict, verbose=False):
    # Set abstract to lower case like concepts
    abstract = abstract.lower()
    if(verbose):
        print(abstract)
        print(ccs)

    # Convert ccs dict into dataframe and order the dataframe by importance
    ccs_df = pd.DataFrame({"ccs_concept": ccs.keys(), "importance": ccs.values()})
    ordered_ccs_array = ccs_df.sort_values("importance")["ccs_concept"].array[::-1]
    # pd.DataFrame({"keywords": count_ccs_kw[ccs_name].keys(), "occurences": count_ccs_kw[ccs_name].values()})

    concept_found = False
    i             = 0
    concept_root  = None
    concept_leaf  = None

    ## Loop that finds a concept root and leaf pair that is occuring in the abstract
    while(not concept_found and i < len(ordered_ccs_array) and ccs[ordered_ccs_array[i]] == 500):
        concept_tree = ordered_ccs_array[i].lower()
        i += 1

        # Identify concept root and leaf
        if('->' in concept_tree):
            concept_list = concept_tree.split("->")
            if(len(concept_list)<=2): continue
            concept_root = concept_list[1]
            concept_leaf = concept_list[-1]
        elif('~' in concept_tree):
            concept_list = concept_tree.split("~")
            if(len(concept_list)<=1): continue
            concept_root = concept_list[0]
            concept_leaf = concept_list[-1]
        else:
            continue

        # Check if the concepts are mentionned in abstract
        if(verbose):
            print("first concept: ", concept_root, "; second concept:", concept_leaf)
            print(concept_root in abstract)
            print(concept_leaf in abstract)
        
        if(concept_root in abstract and concept_leaf in abstract):
            concept_found = True
    
    if(not concept_found):
        return np.nan
    else:
        ## Randomly chooses the order of keywords (kw)
        label = None
        if(np.random.uniform(0,1)>0.5):
            label = 0 # leaf then root is 0
            kw1 = concept_leaf
            kw2 = concept_root

        else:
            label = 1 # root then leaf is 1
            kw2 = concept_leaf
            kw1 = concept_root
            

        ## Computing the distributions of both concepts
        dist1 = ComputeKwDistribution(abstract, kw1)
        dist2 = ComputeKwDistribution(abstract, kw2)

        ## Padding distributions to same length
        len_d1, len_d2 = len(dist1), len(dist2)
        if(len_d1 < len_d2):
            dist1 = np.pad(dist1, (0, len_d2-len_d1), mode="constant", constant_values=0)
        else:
            dist2 = np.pad(dist2, (0, len_d1-len_d2), mode="constant", constant_values=0)

        ## Computing entropy, KL divergence, ratio and Mutual Information for both distributions:
        H1_var = entropy(dist1)
        H2_var = entropy(dist2)
        kl_div1 = kl_div(dist1, dist2).sum()
        kl_div2 = kl_div(dist2, dist1).sum()
        ratio   = kl_div1/kl_div2 if kl_div2 != 0 else np.nan
        MI_var  = mutual_info_score(dist1, dist2) 

        ## Store information
        labeled_data_dict["doi"].append(doi)
        labeled_data_dict["label"].append(label)
        labeled_data_dict["kw1"].append(kw1)
        labeled_data_dict["kw2"].append(kw2)
        labeled_data_dict["distrib1"].append(dist1)
        labeled_data_dict["distrib2"].append(dist2)
        labeled_data_dict["H1"].append(H1_var)
        labeled_data_dict["H2"].append(H2_var)
        labeled_data_dict["DKL_1"].append(kl_div1)
        labeled_data_dict["DKL_2"].append(kl_div2)
        labeled_data_dict["ratio"].append(ratio)
        labeled_data_dict["MI"].append(MI_var)

        return 1