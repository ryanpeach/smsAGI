# SQL Alchemy

Only do a `session.commit()` at the end of the context manager in the same file as the context manager is defined. This way there is no confusion.
