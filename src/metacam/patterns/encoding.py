
import torch
import torch.nn.functional as F
def one_hot_4x4(labels, num_classes=10):
    """
    레이블을 4x4 매트릭스 형태의 One-Hot Encoding으로 변환합니다.

    Args:
        labels (torch.Tensor): 레이블 텐서 (크기: [batch_size]).
        num_classes (int): 클래스 수 (디폴트: 10).

    Returns:
        torch.Tensor: 4x4 매트릭스 형태의 One-Hot Encoding된 레이블 (크기: [batch_size, 4, 4]).
    """
    if not (0 <= labels.max().item() < num_classes):
        raise ValueError("Labels must be in the range [0, 9].")
    
    # 4x4 매트릭스 초기화
    matrix = torch.zeros((labels.size(0), 4, 4))
    
    # 각 숫자에 대해 4x4 매트릭스의 위치 지정
    positions = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1)  # 총 10개 위치
    ]
    
    # 레이블에 맞는 위치에 1을 할당
    for i, label in enumerate(labels):
        row, col = positions[label.item()]
        matrix[i, row, col] = 1
    
    return matrix.unsqueeze(1)


# for the cifar10 dataset
# airplane : 0
# automobile : 1
# bird : 2
# cat : 3
# deer : 4
# dog : 5
# frog : 6
# horse : 7
# ship : 8
# truck : 9
import torch

def one_hot_8x8(labels, num_classes=10):
    """
    레이블을 8x8 매트릭스 형태의 One-Hot Encoding으로 변환합니다.

    Args:
        labels (torch.Tensor): 레이블 텐서 (크기: [batch_size]).
        num_classes (int): 클래스 수 (디폴트: 10).

    Returns:
        torch.Tensor: 8x8 매트릭스 형태의 One-Hot Encoding된 레이블 (크기: [batch_size, 8, 8]).
    """
    if not (0 <= labels.max().item() < num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")
    
    # 8x8 매트릭스 초기화
    matrix = torch.zeros((labels.size(0), 8, 8))
    
    # 각 숫자를 (짝수, 짝수) 위치에 매핑
    # positions = [
    #     (0, 0), (0, 2), (0, 4), (0, 6),
    #     (2, 0), (2, 2), (2, 4), (2, 6),
    #     (4, 0), (4, 2)  # 총 10개 위치
    # ]
    
    positions = [
        (0, 0), (0, 1), (0, 2), (0, 0),
        (0, 0), (4, 2), (4, 2), (4, 2),
        (4, 2), (4, 2)  # 총 10개 위치
    ]
    
    
    # 레이블에 맞는 위치에 1을 할당
    for i, label in enumerate(labels):
        row, col = positions[label.item()]
        matrix[i, row, col] = 1
    
    return matrix.unsqueeze(1)


def one_hot_8x8_to_centor(labels, num_classes=10):
    """
    레이블을 8x8 매트릭스 형태의 One-Hot Encoding으로 변환합니다.

    Args:
        labels (torch.Tensor): 레이블 텐서 (크기: [batch_size]).
        num_classes (int): 클래스 수 (디폴트: 10).

    Returns:
        torch.Tensor: 8x8 매트릭스 형태의 One-Hot Encoding된 레이블 (크기: [batch_size, 8, 8]).
    """
    if not (0 <= labels.max().item() < num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")
    
    # 8x8 매트릭스 초기화
    matrix = torch.zeros((labels.size(0), 8, 8))
    
    # 각 숫자를 (짝수, 짝수) 위치에 매핑
    # positions = [
    #     (0, 0), (0, 2), (0, 4), (0, 6),
    #     (2, 0), (2, 2), (2, 4), (2, 6),
    #     (4, 0), (4, 2)  # 총 10개 위치
    # ]
    
    positions = [
        (4, 4), (4, 4), (4, 4), (4, 4),
          (4, 4), (4, 4), (4, 4), (4, 4), 
          (4, 4), (4, 4)
    ]
    
    
    # 레이블에 맞는 위치에 1을 할당
    for i, label in enumerate(labels):
        row, col = positions[label.item()]
        matrix[i, row, col] = 1
    
    return matrix.unsqueeze(1)



import torch


def one_hot_500x500_to_total(labels, num_classes=10):
    """
    레이블을 500x500 매트릭스 형태의 One-Hot Encoding으로 변환합니다.

    Args:
        labels (torch.Tensor): 레이블 텐서 (크기: [batch_size]).
        num_classes (int): 클래스 수 (디폴트: 10).

    Returns:
        torch.Tensor: 500x500 매트릭스 형태의 One-Hot Encoding된 레이블 (크기: [batch_size, 1, 500, 500]).
    """
    if not (0 <= labels.max().item() < num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")
    
    # 500x500 매트릭스 초기화
    batch_size = labels.size(0)
    matrix = torch.zeros((batch_size, 500, 500), dtype=torch.float32)

    square_size = 70  # One-hot 블록 크기
    width = 500  # 500x500 정사각형
    # 점 패턴 크기 결정 부분
    end_x = 300
    end_y = 300

    positions = []

    # X축을 균등하게 나누기 (3개의 수직 그룹)
    x_values = [0, end_x // 2, end_x]

    # Y축을 균등하게 나누기
    y_values = [0, end_y // 3, (2 * end_y) // 3, end_y]

    # 첫 번째 그룹 (Y축 고정, X = 0)
    for y in y_values:
        positions.append((x_values[0], y))

    # 두 번째 그룹 (Y = 0과 Y = end_y에서 X = end_x//2)
    positions.append((x_values[1], y_values[0]))
    positions.append((x_values[1], y_values[-1]))

    # 세 번째 그룹 (Y축 고정, X = end_x)
    for y in y_values:
        positions.append((x_values[2], y))





    # # 기존 위치 (중앙 정렬 전)
    # positions = [
    #     (0, 0), (0, 80), (0, 160), (0, 240),
    #     (100, 0), (100, 240), (200, 0), (200, 80),
    #     (200, 160), (200, 240)
    # ]

    # width 크기 설정
    

    # 현재 positions의 중심 찾기
    min_x, max_x = min(p[0] for p in positions), max(p[0] for p in positions)
    min_y, max_y = min(p[1] for p in positions), max(p[1] for p in positions)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

    # 새로운 중심 (정사각형 중앙)
    new_center_x, new_center_y = width / 2, width / 2

    # Offset 계산
    offset_x, offset_y = new_center_x - center_x, new_center_y - center_y

    # 중앙으로 이동된 positions (정수 변환)
    positions = [(int(x + offset_x), int(y + offset_y)) for x, y in positions]
    

    # 레이블에 맞는 위치의 정사각형을 1로 채움
    for i, label in enumerate(labels):
        row, col = positions[label.item()]
        
        # 안전한 슬라이싱 (500x500 범위 초과 방지)
        row_end = min(row + square_size, 500)
        col_end = min(col + square_size, 500)

        matrix[i, row:row_end, col:col_end] = 1

    return matrix.unsqueeze(1)  # (batch_size, 1, 500, 500) 형태로 변환




    # positions = [
    #     (0, 0), (0, 150), (0, 300), (0, 450),
    #     (200, 9), (200, 450), (450, 0), (450, 150),
    #         (450,300), (450,450)  # 총 10개 위치
    # ]




# def one_hot_500x500_to_total(labels, num_classes=10):
#     """
#     레이블을 500x500 매트릭스 형태의 One-Hot Encoding으로 변환합니다.

#     Args:
#         labels (torch.Tensor): 레이블 텐서 (크기: [batch_size]).
#         num_classes (int): 클래스 수 (디폴트: 10).

#     Returns:
#         torch.Tensor: 500x500 매트릭스 형태의 One-Hot Encoding된 레이블 (크기: [batch_size, 500, 500]).
#     """
#     if not (0 <= labels.max().item() < num_classes):
#         raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")
    
#     # 500x500 매트릭스 초기화
#     matrix = torch.zeros((labels.size(0), 300, 300))
    
#     # 각 숫자를 매핑할 기준 위치 (짝수, 짝수)
#     positions = [
#         (0, 0), (0, 100), (0, 200), (140, 0),
#         (140, 80), (140, 160), (140, 240), (250, 50),
#             (250,150), (250,250)  # 총 10개 위치
#     ]

#     # positions = [(x // 3, y // 3) for x, y in positions]
#     square_size = 50

#     # 레이블에 맞는 위치의 정사각형을 1로 채움
#     for i, label in enumerate(labels):
#         row, col = positions[label.item()]
#         matrix[i, row:row + square_size, col:col + square_size] = 1
    
#     return matrix.unsqueeze(1)



def one_hot_500x500_to_total_2cols(labels, num_classes=10):
    """
    레이블을 500x500 매트릭스 형태의 One-Hot Encoding으로 변환합니다. (2 columns, 5x2 형태, 이미지 센터 기준 대칭)

    Args:
        labels (torch.Tensor): 레이블 텐서 (크기: [batch_size]).
        num_classes (int): 클래스 수 (디폴트: 10).

    Returns:
        torch.Tensor: 500x500 매트릭스 형태의 One-Hot Encoding된 레이블 (크기: [batch_size, 1, 500, 500]).
    """
    if not (0 <= labels.max().item() < num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")

    batch_size = labels.size(0)
    matrix = torch.zeros((batch_size, 500, 500), dtype=torch.float32)

    square_size = 10
    width = 500
    height = 500

    # 5x2 그리드 설정 (2 columns)
    cols = 2
    rows = 5

    # 블록 간격 계산
    block_width = 50
    block_height = 50

    # 중심 정렬 오프셋 계산
    center_x = width // 2
    center_y = height // 2

    positions = []
    for i in range(rows):
        y_offset = (i - (rows - 1) / 2) * block_height
        for j in range(cols):
            x_offset = (j - (cols - 1) / 2) * block_width * 2
            x = int(center_x + x_offset)
            y = int(center_y + y_offset)
            positions.append((x, y))

    for i, label in enumerate(labels):
        row, col = positions[label.item()]
        row_end = min(row + square_size, 500)
        col_end = min(col + square_size, 500)
        matrix[i, row:row_end, col:col_end] = 1

    return matrix.unsqueeze(1)  # (batch_size, 1, 500, 500) 형태로 변환


def one_hot_500x500_to_total_dl(labels, num_classes=10):
    """
    500x500 매트릭스 형태의 One-Hot Encoding 레이블 생성.

    Args:
        labels (torch.Tensor): 레이블 텐서 (크기: [batch_size]).
        num_classes (int): 클래스 수 (기본값: 10).

    Returns:
        torch.Tensor: One-Hot Encoding된 레이블 (크기: [batch_size, 1, 500, 500]).
    """
    if not (0 <= labels.max().item() < num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")

    batch_size = labels.size(0)
    matrix = torch.zeros((batch_size, 500, 500), dtype=torch.float32)
    square_size = 25
    width = 500
    center_x = center_y = width // 2

    # 기준이 되는 좌표 (300x300 내 균일 분포)
    x_values = [0, 150, 300]
    y_values = [0, 100, 200, 300]
    positions = [(x, y) for x in [0, 300] for y in y_values]  # 양끝 (4개씩)
    positions += [(150, 0), (150, 300)]  # 중간 (위아래 2개)

    # 중앙 정렬을 위해 이동
    offset_x, offset_y = center_x - 150, center_y - 150
    positions = [(x + offset_x, y + offset_y) for x, y in positions]

    # 대칭 적용
    positions += [(2 * center_x - x, y) for x, y in positions]

    # 레이블에 맞는 위치에 정사각형 채우기
    for i, label in enumerate(labels):
        row, col = positions[label.item() % len(positions)]
        row_start, row_end = max(row - square_size // 2, 0), min(row + square_size // 2, width)
        col_start, col_end = max(col - square_size // 2, 0), min(col + square_size // 2, width)
        matrix[i, row_start:row_end, col_start:col_end] = 1

    return matrix.unsqueeze(1)  # (batch_size, 1, 500, 500) 형태



def one_hot_circle_dl(labels, num_classes=10):
    """
    레이블을 500x500 매트릭스 형태의 One-Hot Encoding으로 변환합니다.
    중심을 기준으로 좌우 대칭이 되도록 3-4-3 구조를 유지하며, 간격을 조절할 수 있습니다.

    Args:
        labels (torch.Tensor): 레이블 텐서 (크기: [batch_size]).
        num_classes (int): 클래스 수 (디폴트: 10).
        spacing (int): 픽셀 사이의 간격 (디폴트: 100).

    Returns:
        torch.Tensor: 500x500 매트릭스 형태의 One-Hot Encoding된 레이블 (크기: [batch_size, 1, 500, 500]).
    """
    if not (0 <= labels.max().item() < num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")

    square_size = 15
    width = 1000
    spacing = 100

    center_x = width // 2
    center_y = width // 2

    # 간격에 따라 위치 조정
    x_offsets = [-spacing, 0, spacing]
    y_offsets_top = [-spacing, 0, spacing]
    y_offsets_middle = [-int(1.5 * spacing), -spacing // 2, spacing // 2, int(1.5 * spacing)]

    # 좌측 3개 (Top, Middle, Bottom)
    positions = [(center_x - int(1.5 * spacing), center_y + y) for y in y_offsets_top]

    # 중앙 4개
    positions += [(center_x, center_y + y) for y in y_offsets_middle]

    # 우측 3개 (Top, Middle, Bottom)
    positions += [(center_x + int(1.5 * spacing), center_y + y) for y in y_offsets_top]
    
    
    batch_size = labels.size(0)
    matrix = torch.zeros((batch_size, width, width), dtype=torch.float32)

    # 레이블에 맞는 위치의 정사각형을 1로 채움
    for i, label in enumerate(labels):
        row, col = positions[label.item() % len(positions)]
        row_start = max(row - square_size // 2, 0)
        row_end = min(row + square_size // 2, width)
        col_start = max(col - square_size // 2, 0)
        col_end = min(col + square_size // 2, width)
        matrix[i, row_start:row_end, col_start:col_end] = 1

    return matrix.unsqueeze(1)  # (batch_size, 1, 500, 500) 형태로 변환


def one_hot_circle_dl_30(labels, num_classes=10,square_size=30):
    """
    레이블을 500x500 매트릭스 형태의 One-Hot Encoding으로 변환합니다.
    중심을 기준으로 좌우 대칭이 되도록 3-4-3 구조를 유지하며, 간격을 조절할 수 있습니다.

    Args:
        labels (torch.Tensor): 레이블 텐서 (크기: [batch_size]).
        num_classes (int): 클래스 수 (디폴트: 10).
        spacing (int): 픽셀 사이의 간격 (디폴트: 100).

    Returns:
        torch.Tensor: 500x500 매트릭스 형태의 One-Hot Encoding된 레이블 (크기: [batch_size, 1, 500, 500]).
    """
    if not (0 <= labels.max().item() < num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")

    square_size = square_size
    width = 1000
    spacing = 120

    center_x = width // 2
    center_y = width // 2

    # 간격에 따라 위치 조정
    x_offsets = [-spacing, 0, spacing]
    y_offsets_top = [-spacing, 0, spacing]
    y_offsets_middle = [-int(1.5 * spacing), -spacing // 2, spacing // 2, int(1.5 * spacing)]

    # 좌측 3개 (Top, Middle, Bottom)
    positions = [(center_x - int(1.5 * spacing), center_y + y) for y in y_offsets_top]

    # 중앙 4개
    positions += [(center_x, center_y + y) for y in y_offsets_middle]

    # 우측 3개 (Top, Middle, Bottom)
    positions += [(center_x + int(1.5 * spacing), center_y + y) for y in y_offsets_top]
    
    
    batch_size = labels.size(0)
    matrix = torch.zeros((batch_size, width, width), dtype=torch.float32)

    # 레이블에 맞는 위치의 정사각형을 1로 채움
    for i, label in enumerate(labels):
        row, col = positions[label.item() % len(positions)]
        row_start = max(row - square_size // 2, 0)
        row_end = min(row + square_size // 2, width)
        col_start = max(col - square_size // 2, 0)
        col_end = min(col + square_size // 2, width)
        matrix[i, row_start:row_end, col_start:col_end] = 1

    return matrix.unsqueeze(1)  # (batch_size, 1, 500, 500) 형태로 변환

def one_hot_circle_dl_1980(labels, num_classes=10):
    """
    레이블을 500x500 매트릭스 형태의 One-Hot Encoding으로 변환합니다.
    중심을 기준으로 좌우 대칭이 되도록 3-4-3 구조를 유지하며, 간격을 조절할 수 있습니다.

    Args:
        labels (torch.Tensor): 레이블 텐서 (크기: [batch_size]).
        num_classes (int): 클래스 수 (디폴트: 10).
        spacing (int): 픽셀 사이의 간격 (디폴트: 100).

    Returns:
        torch.Tensor: 500x500 매트릭스 형태의 One-Hot Encoding된 레이블 (크기: [batch_size, 1, 500, 500]).
    """
    if not (0 <= labels.max().item() < num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")
    square_size = 10
    width = 1980
    spacing = 100

    center_x = width // 2
    center_y = width // 2

    # 간격에 따라 위치 조정
    x_offsets = [-spacing, 0, spacing]
    y_offsets_top = [-spacing, 0, spacing]
    y_offsets_middle = [-int(1.5 * spacing), -spacing // 2, spacing // 2, int(1.5 * spacing)]

    # 좌측 3개 (Top, Middle, Bottom)
    positions = [(center_x - int(1.5 * spacing), center_y + y) for y in y_offsets_top]

    # 중앙 4개
    positions += [(center_x, center_y + y) for y in y_offsets_middle]

    # 우측 3개 (Top, Middle, Bottom)
    positions += [(center_x + int(1.5 * spacing), center_y + y) for y in y_offsets_top]

    
    
    batch_size = labels.size(0)
    matrix = torch.zeros((batch_size, width, width), dtype=torch.float32)

    # 레이블에 맞는 위치의 정사각형을 1로 채움
    for i, label in enumerate(labels):
        row, col = positions[label.item() % len(positions)]
        row_start = max(row - square_size // 2, 0)
        row_end = min(row + square_size // 2, width)
        col_start = max(col - square_size // 2, 0)
        col_end = min(col + square_size // 2, width)
        matrix[i, row_start:row_end, col_start:col_end] = 1

    return matrix.unsqueeze(1)  # (batch_size, 1, 500, 500) 형태로 변환


def one_hot_circle_dl_1980_foralign(labels, num_classes=10):
    """
    레이블을 500x500 매트릭스 형태의 One-Hot Encoding으로 변환합니다.
    중심을 기준으로 좌우 대칭이 되도록 3-4-3 구조를 유지하며, 간격을 조절할 수 있습니다.

    Args:
        labels (torch.Tensor): 레이블 텐서 (크기: [batch_size]).
        num_classes (int): 클래스 수 (디폴트: 10).
        spacing (int): 픽셀 사이의 간격 (디폴트: 100).

    Returns:
        torch.Tensor: 500x500 매트릭스 형태의 One-Hot Encoding된 레이블 (크기: [batch_size, 1, 500, 500]).
    """
    if not (0 <= labels.max().item() < num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")
    square_size = 10
    width = 1980
    spacing = 100

    center_x = width // 2
    center_y = width // 2

    # 간격에 따라 위치 조정
    x_offsets = [-spacing, 0, spacing]
    y_offsets_top = [-spacing, 0, spacing]
    y_offsets_middle = [-int(1.5 * spacing), -spacing // 2, spacing // 2, int(1.5 * spacing)]

    positions = []

    # (1) 좌측 맨 위 점만 직접 지정
    positions.append((center_x - int(1.5 * spacing), 840))  # (840, 840)

    # (2) 나머지 좌측 2개 (Middle, Bottom)
    for y in y_offsets_top[1:]:
        positions.append((center_x - int(1.5 * spacing), center_y + y))

    # (3) 중앙 4개
    positions += [(center_x, center_y + y) for y in y_offsets_middle]

    # (4) 우측 3개 (Top, Middle, Bottom)
    positions += [(center_x + int(1.5 * spacing), center_y + y) for y in y_offsets_top]

    
    
    batch_size = labels.size(0)
    matrix = torch.zeros((batch_size, width, width), dtype=torch.float32)

    # 레이블에 맞는 위치의 정사각형을 1로 채움
    for i, label in enumerate(labels):
        row, col = positions[label.item() % len(positions)]
        row_start = max(row - square_size // 2, 0)
        row_end = min(row + square_size // 2, width)
        col_start = max(col - square_size // 2, 0)
        col_end = min(col + square_size // 2, width)
        matrix[i, row_start:row_end, col_start:col_end] = 1

    return matrix.unsqueeze(1)  # (batch_size, 1, 500, 500) 형태로 변환


def one_hot_to_total_dl_1000(labels, num_classes=10):
    """
    500x500 매트릭스 형태의 One-Hot Encoding 레이블 생성.

    Args:
        labels (torch.Tensor): 레이블 텐서 (크기: [batch_size]).
        num_classes (int): 클래스 수 (기본값: 10).

    Returns:
        torch.Tensor: One-Hot Encoding된 레이블 (크기: [batch_size, 1, 500, 500]).
    """
    if not (0 <= labels.max().item() < num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")

    square_size = 10
    width = 1000
    center_x = center_y = width // 2

    # 기준이 되는 좌표 (300x300 내 균일 분포)
    x_values = [0, 150, 300]
    y_values = [0, 100, 200, 300]
    positions = [(x, y) for x in [0, 300] for y in y_values]  # 양끝 (4개씩)
    positions += [(150, 0), (150, 300)]  # 중간 (위아래 2개)

    # 중앙 정렬을 위해 이동
    offset_x, offset_y = center_x - 150, center_y - 150
    positions = [(x + offset_x, y + offset_y) for x, y in positions]

    # 대칭 적용
    positions += [(2 * center_x - x, y) for x, y in positions]    
    
    batch_size = labels.size(0)
    matrix = torch.zeros((batch_size, width, width), dtype=torch.float32)

    # 레이블에 맞는 위치에 정사각형 채우기
    for i, label in enumerate(labels):
        row, col = positions[label.item() % len(positions)]
        row_start, row_end = max(row - square_size // 2, 0), min(row + square_size // 2, width)
        col_start, col_end = max(col - square_size // 2, 0), min(col + square_size // 2, width)
        matrix[i, row_start:row_end, col_start:col_end] = 1

    return matrix.unsqueeze(1)  # (batch_size, 1, 500, 500) 형태

def one_hot_circle_dl_1000(labels, num_classes=10):
    """
    레이블을 500x500 매트릭스 형태의 One-Hot Encoding으로 변환합니다.
    중심을 기준으로 좌우 대칭이 되도록 3-4-3 구조를 유지하며, 간격을 조절할 수 있습니다.

    Args:
        labels (torch.Tensor): 레이블 텐서 (크기: [batch_size]).
        num_classes (int): 클래스 수 (디폴트: 10).
        spacing (int): 픽셀 사이의 간격 (디폴트: 100).

    Returns:
        torch.Tensor: 500x500 매트릭스 형태의 One-Hot Encoding된 레이블 (크기: [batch_size, 1, 500, 500]).
    """
    if not (0 <= labels.max().item() < num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")

    square_size = 15
    width = 1000
    spacing = 100

    center_x = width // 2
    center_y = width // 2

    # 간격에 따라 위치 조정
    x_offsets = [-spacing, 0, spacing]
    y_offsets_top = [-spacing, 0, spacing]
    y_offsets_middle = [-int(1.5 * spacing), -spacing // 2, spacing // 2, int(1.5 * spacing)]

    # 좌측 3개 (Top, Middle, Bottom)
    positions = [(center_x - int(1.5 * spacing), center_y + y) for y in y_offsets_top]

    # 중앙 4개
    positions += [(center_x, center_y + y) for y in y_offsets_middle]

    # 우측 3개 (Top, Middle, Bottom)
    positions += [(center_x + int(1.5 * spacing), center_y + y) for y in y_offsets_top]
    
    
    batch_size = labels.size(0)
    matrix = torch.zeros((batch_size, width, width), dtype=torch.float32)

    # 레이블에 맞는 위치의 정사각형을 1로 채움
    for i, label in enumerate(labels):
        row, col = positions[label.item() % len(positions)]
        row_start = max(row - square_size // 2, 0)
        row_end = min(row + square_size // 2, width)
        col_start = max(col - square_size // 2, 0)
        col_end = min(col + square_size // 2, width)
        matrix[i, row_start:row_end, col_start:col_end] = 1

    return matrix.unsqueeze(1)  # (batch_size, 1, 500, 500) 형태로 변환


def one_hot_circle_dl_500(labels, num_classes=10):
    """
    레이블을 500x500 매트릭스 형태의 One-Hot Encoding으로 변환합니다.
    중심을 기준으로 좌우 대칭이 되도록 3-4-3 구조를 유지하며, 간격을 조절할 수 있습니다.

    Args:
        labels (torch.Tensor): 레이블 텐서 (크기: [batch_size]).
        num_classes (int): 클래스 수 (디폴트: 10).
        spacing (int): 픽셀 사이의 간격 (디폴트: 100).

    Returns:
        torch.Tensor: 500x500 매트릭스 형태의 One-Hot Encoding된 레이블 (크기: [batch_size, 1, 500, 500]).
    """
    if not (0 <= labels.max().item() < num_classes):
        raise ValueError(f"Labels must be in the range [0, {num_classes - 1}].")

    square_size = 15
    width = 500
    spacing = 100

    center_x = width // 2
    center_y = width // 2

    # 간격에 따라 위치 조정
    x_offsets = [-spacing, 0, spacing]
    y_offsets_top = [-spacing, 0, spacing]
    y_offsets_middle = [-int(1.5 * spacing), -spacing // 2, spacing // 2, int(1.5 * spacing)]

    # 좌측 3개 (Top, Middle, Bottom)
    positions = [(center_x - int(1.5 * spacing), center_y + y) for y in y_offsets_top]

    # 중앙 4개
    positions += [(center_x, center_y + y) for y in y_offsets_middle]

    # 우측 3개 (Top, Middle, Bottom)
    positions += [(center_x + int(1.5 * spacing), center_y + y) for y in y_offsets_top]
    
    
    batch_size = labels.size(0)
    matrix = torch.zeros((batch_size, width, width), dtype=torch.float32)

    # 레이블에 맞는 위치의 정사각형을 1로 채움
    for i, label in enumerate(labels):
        row, col = positions[label.item() % len(positions)]
        row_start = max(row - square_size // 2, 0)
        row_end = min(row + square_size // 2, width)
        col_start = max(col - square_size // 2, 0)
        col_end = min(col + square_size // 2, width)
        matrix[i, row_start:row_end, col_start:col_end] = 1

    return matrix.unsqueeze(1)  # (batch_size, 1, 500, 500) 형태로 변환