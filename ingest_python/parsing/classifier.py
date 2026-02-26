from parsing import rules

def classify_line(line: str, source: str) -> str:
    """
    Classifies a line of text for structural parsing using the rule registry.
    
    Args:
        line: The text line to classify.
        source: The document source type (FAR/DFARS/EM385).
        
    Returns:
        The line type identifier (e.g., "PART", "SECTION", "BODY").
    """
    stripped = line.strip()
    if not stripped:
        return "EMPTY"

    # General Footer Checks
    if rules.FOOT_PAGE_NUM.match(stripped):
        return "FOOTER"
    if rules.FOOT_FAR.search(stripped):
        return "FOOTER"
    if rules.FOOT_EM385.search(stripped):
        return "FOOTER"

    # Source-Specific Matching
    if source in ["FAR", "DFARS"]:
        if rules.FAR_PART.match(stripped):
            return "PART"
        if rules.FAR_SUBPART.match(stripped):
            return "SUBPART"
        if rules.FAR_APPENDIX.match(stripped):
            return "APPENDIX"
        if rules.FAR_APPENDIX_PART.match(stripped):
            return "APPENDIX_PART"
        if rules.FAR_APPENDIX_SECTION.match(stripped):
            return "APPENDIX_SECTION"
        if rules.FAR_APPENDIX_RULE.match(stripped):
            return "APPENDIX_SECTION"
        if rules.FAR_SECTION.match(stripped):
            return "SECTION"

    if source == "EM385":
        if rules.EM385_CHAPTER.match(stripped):
            return "CHAPTER"
        if rules.EM385_SECTION.match(stripped):
            return "SECTION"

    return "BODY"
