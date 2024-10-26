import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2


images = np.load('./dataset/images.npy')
labels = np.load('./dataset/labels.npy')

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)


# 인페인팅 함수 정의
def apply_inpainting(image):
    # 이미지가 [0,1] 범위인 경우, 8비트로 변환
    image_uint8 = (image * 255).astype(np.uint8)

    # 하얀 부분(255)을 검출하여 마스크 생성
    mask = cv2.inRange(image_uint8, 255, 255)

    # inpainting을 통해 하얀 부분을 주변 색으로 채움
    image_inpainted = cv2.inpaint(image_uint8, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return image_inpainted


# 훈련 및 테스트 데이터에 대해 인페인팅 적용
train_images = np.array([apply_inpainting(img) for img in train_images])
test_images = np.array([apply_inpainting(img) for img in test_images])


print(train_images.shape) # (816, 28, 28)
print(test_images.shape)  # (204, 28, 28)

conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10


def forward(image, label):
  out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
  out = softmax.forward(out)

  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc


def train(im, label, lr=.005):
  out, loss, acc = forward(im, label)

  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  gradient = softmax.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  gradient = conv.backprop(gradient, lr)

  return loss, acc


train_loss_history = []
train_acc_history = []

for epoch in range(5):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    epoch_loss = 0  # Epoch 전체의 loss 저장
    epoch_num_correct = 0  # Epoch 전체의 정확도 저장

    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, epoch_loss / (i + 1), (epoch_num_correct / (i + 1)) * 100)
            )

        l, acc = train(im, label)
        epoch_loss += l
        epoch_num_correct += acc

    # Epoch 동안의 평균 손실과 정확도를 계산
    avg_loss = epoch_loss / len(train_images)
    avg_acc = epoch_num_correct / len(train_images)

    train_loss_history.append(avg_loss)
    train_acc_history.append(avg_acc)

    print(f'Epoch {epoch + 1} completed: Avg Loss = {avg_loss:.3f}, Avg Accuracy = {avg_acc:.3f}')

print('\n--- Test ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

print(train_loss_history)
print(train_acc_history)

plt.plot(train_loss_history)
plt.plot(train_acc_history)