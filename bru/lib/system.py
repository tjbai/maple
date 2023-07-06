map_template = (
    "Given the following lecture notes or textbook information,"
    " summarize the key points and takeaways that would be relevant"
    " for a student preparing for an exam or reviewing course content."
    " Notes/textbook information: {text}"
    "\nReturn the description in the following format and "
    " limit the response length to around half of the original input."
    "\nsummary name: summary description"
)

reduce_template = (
    "Given the following key points and takeaways,"
    " cohesively summarize all the overarching themes, key points,"
    " and takeaways that would be relevant for a student preparing"
    " for an exam or reviewing course content. Try to condense"
    " \n things to a level of abstraction that makes sense. \n{text}"
)
