
AUTHORIZED_USERS = {1001, 1002, 1005}  # 使用 set 而不是 list，查找 O(1)

def has_access(user_id):
    try:
        user_id = int(user_id) if user_id else None
        return user_id in AUTHORIZED_USERS if user_id else False
    except (ValueError, TypeError):
        return False