import dwhimpalautil
from Project.utils import etl_loop

class ImpalaConnection:
    """
    Base class for creating Impala connection to database
    """

    def __init__(self):
        self.conn = dwhimpalautil.getImpalaConnect()
        self.cursor = self.conn.cursor()
        self.res = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.cursor.close()
        self.conn.close()
        
    def exec_sql(self, sql, is_res_needed=False):
        self.cursor.execute(sql)
        if is_res_needed:
            self.res = self.cursor.fetchall()
            return self.res
        
class ETL(ImpalaConnection):
    """
    Main class for retrieveing data from database
    """    
    def __init__(self):
        super().__init__()

    CREATE_SAMPLE_SQL = """
        create table if not exists usr_erin.churn_sample
        (
                user_id string
            ,   is_churned int
            ,   login_last_dt string
            ,   level int
            ,   donate_total float
        )
    """

    SAMPLE_DATA_SQL = """
        with levels as
        (
            select snap_dt
                ,  cast(user_id as string) as user_id
                ,  max(level) as level
            from game_profiles
            where snap_dt >= '{churned_start_date}'
                and snap_dt < '{churned_end_date}'
            group by snap_dt, user_id
        )
        ,   churned_profiles as
        (
            select a.user_id
                ,  to_date(login_last_ts) as login_last_dt
                ,  1 as is_churned
            from global_profiles 
            where project_id = 1234
                and login_last_ts >= '{churned_start_date}'
                and login_last_ts < '{churned_end_date}'
        )
        ,   not_churned_profiles as
        (
            select user_id
                ,  to_date(date_add('{churned_start_date}', 
                           round((datediff('{churned_end_date}','{churned_start_date}')-1)*rand()))) as login_last_dt
                ,  0 as is_churned 
            from oda_onelink.profiles
            where project_id = 1234
                and login_first_ts < date_sub(cast(unix_timestamp('{churned_start_date}') as timestamp), 30)
                and login_last_ts > date_add(cast(unix_timestamp('{churned_end_date}') as timestamp), 30)
        )
        ,   united as
        (
            select user_id, is_churned, login_last_dt
            from churned_profiles
            union all
            select user_id, is_churned, login_last_dt
            from not_churned_profiles
        )
        ,   payments_total as
        (
            select a.user_id, coalesce(sum(b.pay_amt), 0) as donate_total
            from united a 
                left join payments b
                on (a.user_id = cast(b.global_user_id as string) and a.login_last_dt >= b.log_dt)
            where b.log_dt < '{churned_end_date}'
                and b.project_id in (10,20,30)
            group by a.user_id
        )
        
        insert into usr_erin.churn_sample
        
        select a.user_id
            ,  a.is_churned
            ,  a.login_last_dt
            ,  cast(b.level as int) as level
            ,  c.donate_total
        from united a
            join levels b
            on (a.user_id = b.user_id and a.login_last_dt = b.snap_dt)
            join payments_total c
            on a.user_id = c.user_id
        where level >= {min_level}
            and level <= {max_level}
    """

    CREATE_PROFILES_SQL = """
        create table if not exists usr_erin.churn_profiles 
        (
                user_id string
            ,   age int
            ,   gender string
            ,   reg_country_name string
            ,   days_between_reg_fl int
            ,   days_between_fl_effreg int
            ,   days_between_fl_df int
            ,   has_return_date int
            ,   has_phone_number int
        )
    """

    PROFILES_DATA_SQL = """
        with profiles_data as
        (
            select  a.user_id
                ,   register_ts
                ,   login_first_ts
                ,   return_ts
                ,   donate_first_ts
            from gloabl_profiles a 
                join usr_erin.churn_sample b
                on a.user_id = cast(b.user_id as string)
            where project_id = 1234
        )
        ,   soc_data_raw as
        (
            select  cast(csaid as string) as user_id
                ,   from_unixtime(unix_timestamp(swa_dob ,'yyyyMMdd'), 'yyyy-MM-dd') as birth_date
                ,   swa_gender as gender
                ,   swa_phone as phone
                ,   rank() over (partition by csaid order by created desc) as level
            from soc_data  a 
                join usr_erin.churn_sample b
                on cast(a.csaid as string) = b.user_id
        )
        ,   soc_data as
        (
            select user_id
                ,  birth_date
                ,  gender
                ,  phone
            from soc_data_raw
            where level = 1
        )
        
        insert into usr_erin.churn_profiles
        
        select  a.user_id as user_id
            ,   cast(floor(datediff(now(), birth_date) / 365) as int) as age
            ,   gender
            ,   reg_country_name
            ,   coalesce(datediff(login_first_ts, register_ts), -1) as days_between_reg_fl
            ,   coalesce(datediff(donate_first_ts, login_first_ts), -1) as days_between_fl_df
            ,   if(return_ts is not null, 1, 0) as has_return_date
            ,   if(phone='', 0, 1) as has_phone_number
        from profiles_data a
            left join soc_data b
            on a.user_id = b.user_id
    """
    CREATE_PAYMENTS_SQL = """
        create table if not exists usr_erin.churn_payments
        (
                log_dt string
            ,   user_id string
            ,   pay_amt float
            ,   trans_amt int
        )
    """
    # ........
    CREATE_REPORTS_SQL = """
        create table if not exists usr_erin.churn_reports
        (
                log_dt string
            ,   user_id string
            ,   reports_amt int
        )
    """
    # ........
    CREATE_ABUSERS_SQL = """
        create table if not exists usr_erin.churn_abusers
        (
                log_dt string
            ,   user_id string
            ,   sess_with_abusers_amt int
        )
    """
    # ........
    CREATE_LOGINS_SQL = """
        create table if not exists usr_erin.churn_logins
        (
                log_dt string
            ,   user_id string
            ,   disconnect_amt int
            ,   session_amt int
        )
    """
    # ........
    CREATE_PINGS_SQL = """
        create table if not exists usr_erin.churn_pings
        (
                log_dt string
            ,   user_id string
            ,   avg_min_ping float
        )
    """
    # ........
    CREATE_SESSIONS_SQL = """
        create table if not exists usr_erin.churn_sessions
        (
                log_dt string
            ,   user_id string
            ,   kd float
            ,   win_rate float
            ,   leavings_rate float
            ,   session_player float
        )
    """
    # ........
    CREATE_SHOP_SQL = """
        create table if not exists usr_erin.churn_shop
        (
                log_dt string
            ,   user_id string
            ,   silver_spent string
            ,   gold_spent int
        )
    """
    # ........
    
    TRUNCATE_RAW_DATA_SQL = """
        truncate table {}
    """
    
    CHECK_SQL = """
        select count(*) as cnt, count(distinct user_id) as users
        from {tbl_name}
    """
    
    def create_table(self, sql, is_res_needed=False):
        self.exec_sql(sql, is_res_needed=is_res_needed)
    
    def insert_table(self, sql, is_res_needed=False, **kwargs):
        self.exec_sql(sql.format(**kwargs), is_res_needed=is_res_needed)


@etl_loop
def loading_step(**kwargs):
    """
    Prepare data for one table
    """
    with ETL() as etl:
        tbl = kwargs['table']
        print('Loading -> {}...'.format(tbl))
        etl.create_table(sql=getattr(etl, 'CREATE_{}_SQL'.format(tbl.upper())))
        etl.insert_table(sql=getattr(etl, '{}_DATA_SQL'.format(tbl.upper())), 
                         min_level=kwargs['min_level'],
                         max_level=kwargs['max_level'], 
                         churned_start_date=kwargs['churned_start_date'], 
                         churned_end_date=kwargs['churned_end_date'],
                         data_start_date=kwargs['data_start_date'], 
                         data_end_date=kwargs['data_end_date'])
        etl.insert_table(sql=etl.CHECK_SQL, 
                         tbl_name='usr_erin.churn_{}'.format(tbl), 
                         is_res_needed=True)
        print('Rows = {}, Users = {}'.format(etl.res[0][0], etl.res[0][1]))

@etl_loop
def save_files(table, path):
    """
    Save table data to scv file
    """
    with ETL() as etl:
        print('Saving "{}" into *.csv...'.format(table))
        sql_to_file('select * from usr_erin.churn_{}'.format(table), 
                    '{}{}.csv'.format(path, 'churn_{}'.format(table)))
        
@etl_loop
def load_data(min_level='1',
              max_level='100',
              churned_start_date='2019-01-01', 
              churned_end_date=datetime.strftime(datetime.now()-timedelta(days=30), '%Y-%m-%d'),
              data_start_date='2019-01-01',
              data_end_date=datetime.strftime(datetime.now()-timedelta(days=30), '%Y-%m-%d'), 
              save_to_csv=True, 
              start_with='sample',
              raw_data_path='../input/'):
    """
    Prepare data in database - load important fields
    """
    
    sources = ['sample', 'profiles', 'payments', 'reports', 'abusers', 'logins', 'pings', 'sessions', 'shop']
    start_index = sources.index(start_with)
    
    for tbl in sources[start_index:]:
        loading_step(table=tbl, 
                     min_level=min_level, 
                     max_level=max_level, 
                     churned_start_date=churned_start_date, 
                     churned_end_date=churned_end_date, 
                     data_start_date=data_start_date, 
                     data_end_date=data_end_date)
        if save_to_csv:
            save_files(table=tbl, 
                       path=raw_data_path)
        
    print('Done! All data has been loaded!')