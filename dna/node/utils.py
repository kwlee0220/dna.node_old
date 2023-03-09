
from omegaconf import OmegaConf


def read_node_config(db_conf: OmegaConf, node_id:str) -> OmegaConf:
    from contextlib import closing
    import psycopg2

    with closing(psycopg2.connect(host=db_conf.db_host, dbname=db_conf.db_name,
                                  user=db_conf.db_user, password=db_conf.db_password)) as conn:
        sql = f"select id, camera_conf, tracker_conf, publishing_conf from nodes where id='{node_id}'"
        with closing(conn.cursor()) as cursor:
            cursor.execute(sql)
            row = cursor.fetchone()
            if row is not None:
                import json

                conf = OmegaConf.create()
                conf.id = node_id
                if row[1] is not None:
                    conf.camera = OmegaConf.create(json.loads(row[1]))
                if row[2] is not None:
                    conf.tracker = OmegaConf.create(json.loads(row[2]))
                if row[3] is not None:
                    conf.publishing = OmegaConf.create(json.loads(row[3]))

                return conf
            else:
                return None