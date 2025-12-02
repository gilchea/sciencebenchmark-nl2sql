import logging

logger = logging.getLogger(__name__)

def format_schema(db_id: str, tables_data: list) -> str:
    """Formats the schema for a given db_id into a CREATE TABLE string."""
    db_schema = next((db for db in tables_data if db['db_id'] == db_id), None)
    if not db_schema: return ""

    schema_parts = []
    col_id_to_name = {i: name[1] for i, name in enumerate(db_schema['column_names_original'])}

    for i, table_name in enumerate(db_schema['table_names_original']):
        cols = []
        # Lấy columns thuộc bảng i
        table_cols_indices = [idx for idx, col in enumerate(db_schema['column_names']) if col[0] == i]

        for col_idx in table_cols_indices:
            col_name = col_id_to_name[col_idx]
            col_type = db_schema['column_types'][col_idx]
            if col_name == '*': continue
            cols.append(f"  {col_name} {col_type.upper()}")

        # Primary Keys
        pk_indices = [pk for pk in db_schema['primary_keys'] if pk in table_cols_indices]
        if pk_indices:
            pk_names = ", ".join(col_id_to_name[pk] for pk in pk_indices)
            cols.append(f"  PRIMARY KEY ({pk_names})")

        schema_parts.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(cols) + "\n);")

    # Foreign Keys
    fk_stmts = []
    for f_col, t_col in db_schema['foreign_keys']:
        f_table = db_schema['table_names_original'][db_schema['column_names'][f_col][0]]
        f_name = col_id_to_name[f_col]
        t_table = db_schema['table_names_original'][db_schema['column_names'][t_col][0]]
        t_name = col_id_to_name[t_col]
        fk_stmts.append(f"-- {f_table}.{f_name} TO {t_table}.{t_name}")

    if fk_stmts:
        schema_parts.append("\n-- Foreign Keys:\n" + "\n".join(fk_stmts))

    return "\n".join(schema_parts)

def create_schema_dict(tables_data: list) -> dict:
    return {db['db_id']: format_schema(db['db_id'], tables_data) for db in tables_data}