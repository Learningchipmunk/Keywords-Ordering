import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import nltk
import re

nltk.download('wordnet')
nltk.download('omw-1.4')

stopwords = nltk.corpus.stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()

def findAllIterationsInString(inp_str, sub_str):
      '''Given an input text string (inp_str) and a substring (sub_str), this function
         finds all the positions of the given substring in the input string

      Args:
          inp_str (str): The input string
          sub_str (str): The pattern that needs to be found in the input string

      Returns:
          list: a list of positions of the pattern (sub_str)
      '''
      return [m.start() for m in re.finditer(sub_str, inp_str)]

def stringPreprocessing(inp_str):
      '''Function that preprocesses strings of text. Here are the steps:
            - Removing HTML tags like `<p>`
            - Removing URLs
            - Removing email addresses
            - Removing soft hyphens (\xag)
            - Remove punctuation `,.;@#?!&$()`
            - Lower casing
            - Tokenization
            - Removing Stop words
            - Tokenizetion
            - Lemmatization (to understand meaning in examples)
            - Concatenation and seperation by spaces

      Args:
          inp_str (string): the string of text that is to be preprocessed

      Returns:
          output_str: preprocessed string
      '''
      ## Removing HTML tags
      no_tags = re.sub(r"""
            <.*?>         # Accept HTML tags
            """,
            " ",          # and replaces it with a single space
            inp_str, flags=re.VERBOSE)

      ## Removing URLs
      no_URLs = re.sub(r"""
            (DOI: )?https?:\/\/.*[\r\n]*  # Accepts URLs that could be preceeded by DOI:
            """, 
            " ",                          # and replaces it with a single space
            no_tags, flags=re.VERBOSE)

      ## Removing email addresses
      no_emails = re.sub(r"""
            \S*@\S*\s*        # Match email addresses with every possible character
            """, 
            " ",              # and replaces it with a space
            no_URLs, flags=re.VERBOSE)

      ## Removing Soft Hyphens (\xad)
      no_SH = re.sub(r"""
            \xad *            # Accepts Soft Hyphens
            """, 
            "",               # and removes it from the string
            no_emails, flags=re.VERBOSE)

      ## Removing punctuation
      no_punct = re.sub(r"""
            [,.:;@#?!&$()|]+  # Accept one or more copies of punctuation
            \ *               # plus zero or more copies of a space,
            """,
            " ",              # and replaces it with a single space
            no_SH, flags=re.VERBOSE)

      ## Lowering
      lowered = no_punct.lower()

      ## Tokenization
      tokenized = lowered.split()

      ## removing stop words
      no_stpwrds = [i for i in tokenized if i not in stopwords]

      ## Lemmatization
      lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in no_stpwrds]

      ## Concatenation
      output_str = " ".join(lemm_text)

      return output_str

def ChooseMostImportantConcept(ccs):
      '''Function that loops over concept dict and chooses the most important concept

      Args:
          ccs (dict(key:str,value:int))): a concept dict made of CCS concept branches as keys and importance level 
                      as values (500-300-100). The highest the value, the more important is the concept.
                      E.g. ccs['CCS->Software and its engineering->Software notations and tools
                              ->General programming languages->Language type']
                              returns 500, meaning that is a concept of High importance

      Returns:
          string: the concept branch that is the most important
      '''
      max_importance = 0
      associated_concept = np.nan
      for key, value in ccs.items():
            if value > max_importance:
                  max_importance = value
                  associated_concept = key
            
      return associated_concept

def ChooseMostImportantConcepts(ccs):
      '''Same reasoning as in ChooseMostImportantConcept, but here we choose all of them rather than one

      Args:
          ccs (dict): same desc as ChooseMostImportantConcept

      Returns:
          list: list of CCS branches
      '''
      if(list(ccs.values()) == []): return []
      max_importance = np.max(list(ccs.values()))
      associated_concepts = []
      for key, value in ccs.items():
            if value >= max_importance:
                  associated_concepts.append(key)
            
      return associated_concepts

def ExtractRootAndLeafFromCCSBranch(ccs_branch, preprocess=False):
      concept_root, concept_leaf = None, None
      if('->' in ccs_branch):
            concept_list = ccs_branch.split("->")
            if(len(concept_list)<=2): return np.nan, np.nan
            concept_root = concept_list[1]
            concept_leaf = concept_list[-1]
      elif('~' in ccs_branch):
            concept_list = ccs_branch.split("~")
            if(len(concept_list)<=1): return np.nan, np.nan
            concept_root = concept_list[0]
            concept_leaf = concept_list[-1]
      else:
            return np.nan, np.nan

      if preprocess:
            concept_root, concept_leaf = stringPreprocessing(concept_root), stringPreprocessing(concept_leaf)

      return concept_root, concept_leaf

## Define a function that takes the ccs dict and returns main root main leaf and main ccs branches
def ExtractMainRootLeafAndConcepts(ccs):
      '''Function that takes a ccs dictionnary and returns three elements:
         the main concept root
         the main concept leaf
         and all the concept root concept leaf pairs of high importance [(concept_root, concept_leaf), ...]
         Example:
            input: {'CCS->Mathematics of computing->Mathematical analysis->Functional analysis->Approximation': 100, 
                  'CCS->Theory of computation->Design and analysis of algorithms->Approximation algorithms analysis': 100,
                  'CCS->Mathematics of computing->Mathematical analysis->Numerical analysis->Computations on matrices': 500, 
                  'CCS->Computing methodologies->Symbolic and algebraic manipulation->Symbolic and algebraic algorithms->Linear algebra algorithms': 500}
            Output: ('mathematics computing',
                  'computation matrix',
                  [('mathematics computing', 'computation matrix'),
                  ('computing methodology', 'linear algebra algorithm')])

      Args:
          ccs (dict): {'CCS BRanch':Importance(integer)}

      Returns:
          list: list of three elements
      '''
      associated_concept_list = ChooseMostImportantConcepts(ccs)
      if(associated_concept_list == []):return [np.nan, np.nan, np.nan]
      high_importance_concept_leafs_and_roots = [ExtractRootAndLeafFromCCSBranch(branch, preprocess=True) for branch in associated_concept_list]
      concept_root, concept_leaf = high_importance_concept_leafs_and_roots[0]
      return [concept_root, concept_leaf, high_importance_concept_leafs_and_roots]