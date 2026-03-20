import re

# ---------------------------------------------------------
# General Footer Rules
# ---------------------------------------------------------
FOOT_PAGE_NUM = re.compile(r"^\d+$")
FOOT_FAR = re.compile(r"FEDERAL ACQUISITION REGULATION", re.IGNORECASE)
FOOT_EM385 = re.compile(r"EM\s+385")

# ---------------------------------------------------------
# FAR / DFARS Specific Rules
# ---------------------------------------------------------
FAR_PART = re.compile(r"^PART\s+\d+\b", re.IGNORECASE)
FAR_SUBPART = re.compile(r"^Subpart\s+\d+\.\d+\b", re.IGNORECASE)
FAR_APPENDIX = re.compile(r"^APPENDIX\s+[A-Z]\b", re.IGNORECASE)
FAR_APPENDIX_PART = re.compile(r"^Part\s+\d+\b", re.IGNORECASE)
FAR_APPENDIX_SECTION = re.compile(r"^[A-Z]-\d+\b")
FAR_APPENDIX_RULE = re.compile(r"^Rule\s+\d+\b", re.IGNORECASE)
FAR_SECTION = re.compile(r"^\d{2,3}\.\d{3}(?:-\d+)?(?:\([a-z]\)(?:\(\d+\))?)?\s+")

# ---------------------------------------------------------
# EM385 Specific Rules
# ---------------------------------------------------------
EM385_CHAPTER = re.compile(r"^Chapter\s+\d+\b", re.IGNORECASE)
EM385_SECTION = re.compile(r"^(?:\d+-\d+\.|\d{2}\.[A-Z]\.\d{2})\s+")
